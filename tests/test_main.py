from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from fdua_competition.main import get_root, get_documents_dir, get_queries, get_pages, get_vectorstore, get_prompt, get_chat_model


def test_document_dir():
    assert get_documents_dir().name == "documents"


def test_get_pages():
    pages = list(get_pages("1.pdf"))
    assert len(pages) > 0


def test_get_vectorstore():
    vectorstore = get_vectorstore(
        model="embedding",
        embedding_class=AzureOpenAIEmbeddings,
        vectorstore_class=InMemoryVectorStore,
    )
    assert vectorstore


def test_prompt():
    system_prompt = "test system prompt: {language}"
    query = "test query"
    vectorstore = get_vectorstore(
        model="embedding",
        embedding_class=AzureOpenAIEmbeddings,
        vectorstore_class=InMemoryVectorStore,
    )
    prompt = get_prompt(system_prompt, query, vectorstore)
    assert prompt


def test_get_chat_model():
    chat_model = get_chat_model()
    assert chat_model


def test_get_queries():
    queries = get_queries()
    assert isinstance(queries, list)
    assert queries
