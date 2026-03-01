import os

from dotenv import load_dotenv
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()
console = Console()

# 1. Google API 키 설정 (Google AI Studio에서 발급 가능)
api_key = os.getenv("GOOGLE_API_KEY")

def print_rich_response(response_text):
    # 제미나이가 준 답변(Markdown)을 터미널에 렌더링해서 출력합니다.
    md = Markdown(response_text)
    console.print(md)

def gemini_rag_system(pdf_path, query):
    # STEP 1: PDF 로드
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # STEP 2: 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # STEP 3: 제미나이 전용 임베딩 사용
    # 모델명: models/embedding-001
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # STEP 4: 제미나이 LLM 설정
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # STEP 5: 검색기 및 QA 체인 실행
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    return result


def get_relevant_documents_paths(data_path, query, k):
    # 1. 환경 설정
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    DB_PATH = "faiss_index"

    # 2. 벡터 DB 로드 또는 생성 (기존 로직 유지)
    if os.path.exists(DB_PATH):
        print(f"로그: '{DB_PATH}'에서 기존 벡터 DB를 불러옵니다...")
        vectorstore = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print(f"로그: 새 벡터 DB를 생성합니다...")
        loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(DB_PATH)

    # 3. 유사도 검색 수행 (Similarity Search)
    # 질문과 가장 가까운 k개의 문서 조각(Document 객체) 불러오기
    relevant_docs = vectorstore.similarity_search(query, k=k)

    # 4. 파일 경로 추출 및 중복 제거
    # 각 문서 조각의 metadata['source']에 파일 전체 경로 적재
    # set을 사용하여 중복을 제거한 뒤 리스트로 변환
    path_list = list(set([doc.metadata['source'] for doc in relevant_docs]))

    return path_list