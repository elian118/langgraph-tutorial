from langchain_community.document_loaders import PyMuPDFLoader

from naive_rag.utils import gemini_rag_system, print_rich_response, get_relevant_documents_paths

pdf_path = "data/삼성전자, MWC26에서 갤럭시 AI 경험과 기술 혁신 선보여.pdf"
data_path = "data/"
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

def test_load_pdf():
    print(f"총 {len(docs)} 페이지를 불러왔습니다.")
    print(docs[0].page_content[:100])  # 1페이지 앞부분 출력

def test_rag_system(query):
    result = gemini_rag_system(pdf_path, query)
    print("\n" + "=" * 50)
    print(f"질문: {query}")
    print("-" * 50)

    print_rich_response(result['result'])
    print("=" * 50 + "\n")

def test_multidoc_rag_system(query):
    result = get_relevant_documents_paths(data_path, query, 10)
    print("\n" + "=" * 50)
    print(f"질문: {query}")
    print("-" * 50)

    print_rich_response(result['result'])
    print("=" * 50 + "\n")

if __name__ == "__main__":
    test_multidoc_rag_system("삼성전자가 밀라노 동계올림픽에서 무엇을 후원했지?")
