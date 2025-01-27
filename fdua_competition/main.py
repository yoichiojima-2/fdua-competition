import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_core.documents import Document
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.vectorstores.base import VectorStore, VectorStoreRetriever
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


# [START: paths]
def get_root() -> Path:
    return Path().home() / ".fdua-competition"


def get_documents_dir() -> Path:
    return get_root() / "downloads/documents"


def get_output_path() -> Path:
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M')}.md"


# [END: paths]


# [START: queries]
@traceable
def get_queries() -> list[str]:
    df = pd.read_csv(get_root() / "downloads/query.csv")
    return df["problem"].tolist()


# [END: queries]


# [START: vectorstores]
@traceable
def get_pages(filename: str) -> Iterable[Document]:
    pdf_path = get_documents_dir() / filename
    for doc in PyPDFium2Loader(pdf_path).lazy_load():
        yield doc


@traceable
def get_embedding_model(opt: str) -> OpenAIEmbeddings:
    match opt:
        case "azure":
            return AzureOpenAIEmbeddings(azure_deployment="embedding")
        case _:
            raise ValueError(f"unknown model: {opt}")


def get_vectorstore(opt: str, embeddings: OpenAIEmbeddings) -> VectorStore:
    match opt:
        case "in-memory":
            return InMemoryVectorStore(embeddings)
        case "chroma":
            return Chroma(
                collection_name="fdua-competition",
                embedding_function=embeddings,
                persist_directory=str(get_root() / "vectorstore/chroma"),
            )
        case _:
            raise ValueError(f"unknown vectorstore: {opt}")


@traceable
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
def add_documents_with_retry(vectorstore: VectorStore, batch: list[Document]) -> None:
    vectorstore.add_documents(batch)


@traceable
def add_pages_to_vectorstore_in_batches(vectorstore: VectorStore, pages: Iterable[Document], batch_size: int = 8) -> None:
    batch = []
    for page in tqdm(pages, desc="adding pages.."):
        batch.append(page)
        if len(batch) == batch_size:
            add_documents_with_retry(vectorstore=vectorstore, batch=batch)
            batch = []  # clear the batch

    # add any remaining documents in the last batch
    if batch:
        add_documents_with_retry(vectorstore=vectorstore, batch=batch)


@traceable
def add_document_to_vectorstore(vectorstore: VectorStore) -> VectorStore:
    for path in get_documents_dir().glob("*.pdf"):
        print(f"adding document to vectorstore: {path}")
        add_pages_to_vectorstore_in_batches(vectorstore=vectorstore, pages=get_pages(path))


# [END: vectorstores]


# [START: chat]


@dataclass
class GetPromptResult:
    value: ChatPromptValue
    metadata: list[dict]


def get_prompt(system_prompt: str, query: str, retriever: VectorStoreRetriever, language: str = "Japanese") -> GetPromptResult:
    relevant_pages = retriever.invoke(query)

    context = [page.page_content for page in relevant_pages]
    metadata = [page.metadata for page in relevant_pages]

    value = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "context: {context}"),
            ("user", "query: {query}"),
        ]
    ).invoke(
        {
            "language": language,
            "context": "\n".join(context),
            "query": query,
        }
    )

    return GetPromptResult(value=value, metadata=metadata)


@traceable
def get_chat_model(opt: str) -> ChatOpenAI:
    match opt:
        case "azure":
            return AzureChatOpenAI(azure_deployment="4omini")
        case _:
            raise ValueError(f"unknown model: {opt}")


# [END: chat]


def parse_metadata(metadata: list[dict]) -> str:
    return ",  ".join([f"p{data['page']} - {Path(data['source']).name}" for data in metadata])


@traceable
def main() -> None:
    chat_model = get_chat_model("azure")
    system_prompt = "Answer the following question based only on the provided context in {language}"

    embedding_model = get_embedding_model("azure")
    retriever = get_vectorstore("chroma", embedding_model).as_retriever()
    # add_document_to_vectorstore(vectorstore)

    with get_output_path().open(mode="a") as f:
        f.write("# Results\n")

        for query in tqdm(get_queries(), desc="querying.."):
            prompt = get_prompt(system_prompt, query, retriever)
            res = chat_model.invoke(prompt.value)

            print(f"query: {query}")
            f.write(f"## {query}\n")

            print(f"output: {res.content}")
            f.write(f"{res.content}\n")

            parsed_metadata = parse_metadata(prompt.metadata)
            print(f"source: {parsed_metadata}")
            f.write(f"source: {parsed_metadata}\n")

            print()
            f.write("\n")


if __name__ == "__main__":
    load_dotenv()
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
