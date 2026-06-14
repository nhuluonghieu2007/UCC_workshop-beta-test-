import os
import glob
from typing import List
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# LangChain & LangGraph
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Vector DB
import chromadb
from chromadb.utils import embedding_functions

# 1. CẤU HÌNH BIẾN MÔI TRƯỜNG
load_dotenv()

# 2. KẾT NỐI CHROMA CLOUD
client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CHROMA_API_KEY")
)

# Sử dụng mô hình đa ngôn ngữ để hỗ trợ tiếng Việt tốt nhất
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection_name = "workshop_multi_pdf_rag"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=sentence_transformer_ef
)

# 3. HÀM XỬ LÝ PDF (INGESTION)
def ingest_pdf(pdf_path: str):
    """Xử lý từng file PDF: Load -> Split -> Upsert[cite: 4, 31]."""
    try:
        print(f"📄 Đang xử lý file: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Structural Chunking: Sử dụng Regex để ngắt theo Chương/Điều [cite: 50, 52]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=[
                r"\nChương\s+[IVXLCDM]+", 
                r"\nĐiều\s+\d+",             
                r"\n\d+\.\s",                
                r"\n",                       
                r"\s+"
            ],
            is_separator_regex=True
        )
        chunks = splitter.split_documents(pages)

        # Batch Upsert để tối ưu hóa hiệu suất mạng [cite: 14]
        collection.upsert(
            ids=[f"{os.path.basename(pdf_path)}_chunk_{i}" for i in range(len(chunks))],
            documents=[c.page_content for c in chunks],
            metadatas=[{"source": pdf_path, "page": c.metadata.get("page", 0)} for c in chunks],
        )
        print(f"✅ Đã nạp {len(chunks)} chunks từ {os.path.basename(pdf_path)}")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý file {pdf_path}: {e}")

def ingest_folder(folder_path: str):
    """Quét thư mục và nạp nhiều file PDF bằng đa luồng (Cấp độ 2)[cite: 7, 12]."""
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ Không tìm thấy file PDF nào trong: {folder_path}")
        return

    print(f"🚀 Tìm thấy {len(pdf_files)} file. Đang nạp đa luồng (max_workers=4)...")
    
    # Sử dụng ThreadPoolExecutor để chạy song song 
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(ingest_pdf, pdf_files)
    
    print("✨ Hoàn thành nạp toàn bộ thư mục!")

# 4. CÔNG CỤ TRUY VẤN (RETRIEVAL TOOL)
@tool
def retrieve_legal_docs(query: str) -> str:
    """
    Tìm kiếm thông tin trong cơ sở dữ liệu Luật.
    QUY TẮC: Truyền nguyên văn câu hỏi tiếng Việt của người dùng.
    """
    results = collection.query(query_texts=[query], n_results=5)
    passages = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not passages:
        return "Không tìm thấy thông tin liên quan trong tài liệu."

    formatted = []
    for i, (text, meta) in enumerate(zip(passages, metadatas), 1):
        source = os.path.basename(meta.get("source", "unknown"))
        page = meta.get("page", "?")
        formatted.append(f"[Đoạn {i} - Nguồn: {source}, Trang: {page}]\n{text}")

    return "\n\n---\n\n".join(formatted)

# 5. CẤU HÌNH AGENT & LANGGRAPH
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên gia về Luật dân số và pháp luật Việt Nam.
1. Luôn dùng công cụ 'retrieve_legal_docs' để tìm dữ liệu trước khi trả lời.
2. Trả lời dựa CHÍNH XÁC trên ngữ cảnh được cung cấp.
3. Luôn trích dẫn Nguồn và Số trang ở cuối câu trả lời.
4. Nếu không có trong tài liệu, hãy nói 'Tôi không tìm thấy thông tin này'."""

tools = [retrieve_legal_docs]
# Sử dụng model Llama 3.3 70B (hoặc model Groq bạn có quyền truy cập)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(tools)

def agent_node(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Xây dựng đồ thị (Graph)
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

# 6. CHƯƠNG TRÌNH CHÍNH (MAIN LOOP)
if __name__ == "__main__":
    print("\n--- RAG AGENT READY (FOLDER SUPPORT) ---")
    print("Gõ 'ingest <path>' để nạp 1 file")
    print("Gõ 'ingest_folder <path>' để nạp cả thư mục [cite: 8]")
    print("Gõ 'quit' để thoát\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit": break
        
        # Xử lý lệnh nạp folder
        if user_input.lower().startswith("ingest_folder "):
            path = user_input[14:].strip()
            if os.path.isdir(path): ingest_folder(path)
            else: print("❌ Thư mục không tồn tại!")
            continue

        # Xử lý lệnh nạp file
        if user_input.lower().startswith("ingest "):
            path = user_input[7:].strip()
            if os.path.exists(path): ingest_pdf(path)
            else: print("❌ File không tồn tại!")
            continue

        # Chat bình thường
        history.append(HumanMessage(content=user_input))
        result = graph.invoke({"messages": history})
        history = result["messages"]
        
        print(f"\nAgent: {history[-1].content}\n")