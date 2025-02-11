"""
vectorstoreの構築
"""

from pathlib import Path
from pprint import pprint
import typing as t
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random
import textwrap
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re

from fdua_competition.enums import EmbeddingModelOption, Mode, VectorStoreOption
from fdua_competition.utils import get_root, print_before_retry

def before_sleep_hook(state) -> None:
    print(f":( retrying attempt {state.attempt_number} after exception: {state.outcome.exception()}")

class CleansePDF(BaseModel):
    output: str = Field(description="The cleansed 'response' string that satisfies the requirements.")


def split_document(doc: Document) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=10,
        separators=["\n\n", "\n", "。", "．", "？", "！", "「", "」", "【", "】"],
    )
    split_doc = splitter.split_text(doc.page_content)
    return [Document(page_content=d, metadata=doc.metadata) for d in split_doc]

def remove_special_characters(doc: Document) -> Document:
    # remove control characters
    pattern = r"[\x00-\x08\x0B-\x0C\x0E-\x1F]"
    return Document(page_content=re.sub(pattern, "", doc.page_content), metadata=doc.metadata)


@retry(stop=stop_after_attempt(24), wait=wait_random(min=0, max=8), before_sleep=before_sleep_hook)
def cleanse_pdf(doc: Document) -> CleansePDF:
    role = textwrap.dedent(
        """
        You are an intelligent assistant specializing in text refinement.
        The input provided is raw data parsed from a PDF and may be messy or contain unwanted artifacts.
        Your task is to clean up this raw context with only minimal modifications, ensuring that no important information is lost.

        ## Instructions:
        - Fix minor formatting issues (such as extra whitespace, punctuation errors, or unwanted artifacts) without removing any essential content.
        - Do not rephrase or add new information.
        - Preserve all critical details while cleaning the text.
        - The final output must be a concise
        - Do not use commas or special characters that may break JSON parsing.

        ## Input:
        - **context**: The raw context data extracted from a PDF.

        ## Output:
        Return the cleaned context text with minimal corrections, preserving all original information.
        """
    )
    chat_model = AzureChatOpenAI(azure_deployment="4omini").with_structured_output(CleansePDF)
    prompt_template = ChatPromptTemplate.from_messages([("system", role), ("user", "input: {input}")])
    chain = prompt_template | chat_model

    docs = split_document(doc)
    cleansed_text = "".join([chain.invoke({"input": remove_special_characters(doc)}).output for doc in docs])
    res = CleansePDF(input=doc.page_content, output=cleansed_text)
    # print(f"[cleanse_pdf] done\n{dict_to_yaml(res.model_dump())}\n")
    print(res)
    return res


def get_documents_dir(mode: Mode) -> Path:
    """
    指定されたモードに基づいてPDFが入っているパスを取得する
    args:
        mode (Mode): 動作モード(TEST または SUBMIT)
    returns:
        Path: PDFが入っているパス
    raises:
        ValueError: 未知のモードが指定された場合
    """
    match mode:
        case Mode.TEST:
            return get_root() / "validation/documents"
            # return get_root() / "validation/test_doc"

        case Mode.SUBMIT:
            return get_root() / "documents"

        case _:
            raise ValueError(f"): unknown mode: {mode}")


def get_document_list(document_dir: Path) -> list[Path]:
    """
    指定されたディレクトリ内の PDF ファイルのパスのリストを取得する
    args:
        document_dir (Path): Documentディレクトリのパス
    returns:
        list[Path]: PDF ファイルのパスのリスト
    """
    return [path for path in document_dir.glob("*.pdf")]

def split_document(doc: Document) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20000,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "．", "？", "！", "「", "」", "【", "】"],
    )
    split_doc = splitter.split_text(doc.page_content)
    return [Document(page_content=d, metadata=doc.metadata) for d in split_doc]


def load_pages(path: Path) -> t.Iterable[Document]:
    """
    PDF ファイルからページごとのDocument(Document)を読み込むジェネレーター
    args:
        path (Path): PDF ファイルのパス
    yields:
        Document: 読み込まれたDocumentページ
    """
    # for doc in PyPDFium2Loader(path).lazy_load():
    docs = list(PyPDFium2Loader(path).lazy_load())
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=20000,  # チャンクのサイズ (文字数)
    #     chunk_overlap=0,  # チャンクの重複サイズ (文字数)
    # )
    # print(type(docs[0]))
    # # pprint(doc)
    # pprint(docs[0])
    

    # documents = [Document(page_content=doc.page_content)]
    # documents = [i.page_content for i in doc]
    # print(type(documents[0]))
    # chunks = text_splitter.split_documents([docs[0]])
    # chunks = text_splitter.create_documents(docs[0].page_content)
    chunk_list = []
    for doc in docs:
        # chunk_list.extend(split_document(doc))
        chunk_list.extend([
            Document(metadata = doc.metadata, page_content = cleanse_pdf(doc).output)
        ])
    
    pprint(chunk_list)
    pprint(type(chunk_list[0]))
    chunks = chunk_list
    # chunks = text_splitter.split_documents(doc)
    # yield chunks
    return chunks


def get_embedding_model(opt: EmbeddingModelOption) -> OpenAIEmbeddings:
    """
    指定されたembeddingモデルオプションに基づいてembeddingモデルを取得する
    args:
        opt (EmbeddingModelOption): embeddingモデルのオプション
    returns:
        OpenAIEmbeddings: embeddingモデルのインスタンス
    raises:
        ValueError: 未知のモデルオプションが指定された場合
    """
    match opt:
        case EmbeddingModelOption.AZURE:
            return AzureOpenAIEmbeddings(azure_deployment="embedding")
        case _:
            raise ValueError(f"): unknown model: {opt}")


def prepare_vectorstore(output_name: str, opt: VectorStoreOption, embeddings: OpenAIEmbeddings) -> VectorStore:
    """
    指定されたパラメータに基づいてvectorstoreを準備する
    args:
        output_name (str): vectorstoreのコレクション名
        opt (VectorStoreOption): vectorstoreのオプション
        embeddings (OpenAIEmbeddings): embeddingモデルのインスタンス
    returns:
        VectorStore: 構築されたvectorstore
    raises:
        ValueError: 未知のvectorstoreオプションが指定された場合
    """
    match opt:
        case VectorStoreOption.IN_MEMORY:
            return InMemoryVectorStore(embeddings)

        case VectorStoreOption.CHROMA:
            persist_directory = get_root() / "vectorstores/chroma"
            print(f"[prepare_vectorstore] chroma: {persist_directory}")
            persist_directory.mkdir(parents=True, exist_ok=True)
            return Chroma(
                collection_name=output_name,
                embedding_function=embeddings,
                persist_directory=str(persist_directory),
            )

        case _:
            raise ValueError(f"): unknown vectorstore: {opt}")


def _get_existing_sources_in_vectorstore(vectorstore: VectorStore) -> set[str]:
    """
    vectorstore内に既に登録されているDocumentのソース一覧を取得する
    args:
        vectorstore (VectorStore): vectorstoreのインスタンス
    returns:
        set[str]: 登録済みDocumentのソースの集合
    """
    return {metadata.get("source") for metadata in vectorstore.get().get("metadatas")}


@retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=print_before_retry)
def _add_documents_with_retry(vectorstore: VectorStore, batch: list[Document]) -> None:
    """
    リトライ機能付きでDocumentのバッチをvectorstoreに追加する
    args:
        vectorstore (VectorStore): vectorstoreのインスタンス
        batch (list[Document]): 追加するDocumentのバッチ
    """
    vectorstore.add_documents(batch)


def _add_pages_to_vectorstore_in_batches(vectorstore: VectorStore, pages: t.Iterable[Document], batch_size: int = 8) -> None:
    """
    Documentページをバッチごとにvectorstoreへ追加する
    args:
        vectorstore (VectorStore): vectorstoreのインスタンス
        pages (Iterable[Document]): 追加するDocumentページのIterable
        batch_size (int, optional): バッチサイズ (デフォルトは8)
    """
    batch = []
    # print(pages)
    print(type(pages))
    for page in tqdm(pages, desc="adding pages.."):
        batch.append(page)

        if len(batch) == batch_size:
            _add_documents_with_retry(vectorstore=vectorstore, batch=batch)
            batch = []

    if batch:
        _add_documents_with_retry(vectorstore=vectorstore, batch=batch)


def add_documents_to_vectorstore(documents: list[Path], vectorstore: VectorStore) -> None:
    """
    指定された PDF ファイル群をvectorstoreに追加する
    既に登録されているDocumentはスキップされる
    args:
        documents (list[Path]): PDF ファイルのパスのリスト
        vectorstore (VectorStore): vectorstoreのインスタンス
    """
    existing_sources = _get_existing_sources_in_vectorstore(vectorstore)

    for path in documents:
        # if str(path) in existing_sources:
        #     print(f"[add_document_to_vectorstore] skipping existing document: {path}")
        #     continue

        print(f"[add_document_to_vectorstore] adding document to vectorstore: {path}")
        pages = load_pages(path=path)
        _add_pages_to_vectorstore_in_batches(vectorstore=vectorstore, pages=pages)


def build_vectorstore(output_name: str, mode: Mode, vectorstore_option: VectorStoreOption) -> VectorStore:
    """
    指定されたパラメータに基づいてvectorstoreを構築する
    PDF ファイルを読み込み, Documentをvectorstoreに追加する
    args:
        output_name (str): vectorstoreのコレクション名
        mode (Mode): 動作モード(TEST または SUBMIT)
        vectorstore_option (VectorStoreOption): 使用するvectorstoreのオプション
    returns:
        VectorStore: 構築されたvectorstore
    """
    embeddings = get_embedding_model(EmbeddingModelOption.AZURE)
    vectorstore = prepare_vectorstore(output_name=output_name, opt=vectorstore_option, embeddings=embeddings)
    docs = get_document_list(document_dir=get_documents_dir(mode=mode))
    pprint(docs)
    add_documents_to_vectorstore(docs, vectorstore)

    return vectorstore


@retry(stop=stop_after_attempt(24), wait=wait_fixed(1), before_sleep=print_before_retry)
def retrieve_context(vectorstore: VectorStore, query: str) -> str:
    """
    指定されたクエリに対して, 関連Documentから文脈を構築する
    args:
        vectorstore (VectorStore): vectorstoreのインスタンス
        query (str): クエリ
    returns:
        str: 構築された文脈情報
    """
    pages = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}).invoke(query)

    # 各ページの内容とメタデータを整形して文脈として連結
    contexts = ["\n".join([f"page_content: {page.page_content}", f"metadata: {page.metadata}"]) for page in pages]

    return "\n---\n".join(contexts)
