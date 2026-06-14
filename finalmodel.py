# demo5_rag_agent.py
import os
import concurrent.futures # BỔ SUNG: Thư viện để xử lý đa luồng nạp nhiều file cùng lúc
from dotenv import load_dotenv
#from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ─────────────────────────────────────────
# 1. CONNECT TO YOUR CHROMA CLOUD COLLECTION
# ─────────────────────────────────────────
client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CHROMA_API_KEY")
)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
collection_name = "luat_dan_so_vietnam8"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=sentence_transformer_ef
)
print(f" Đã tạo/kết nối thành công Collection: '{collection_name}'")

# ─────────────────────────────────────────
# 2. PDF INGESTION HELPER  
# ─────────────────────────────────────────
def ingest_pdf(pdf_path: str):
    """Load a PDF, split into chunks, and upsert into ChromaDB."""
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    # =================================================================
    # GHI CHÚ TỪ TÀI LIỆU VỀ CÁC CHIẾN LƯỢC CHUNKING (TACTIC CHUNKING):
    # Ở đây chúng ta đang kết hợp 2 phương pháp:
    # 1. 2.1 Cắt đệ quy theo ký tự (Recursive Character Chunking): Cố gắng cắt theo đoạn văn (\n\n), rồi mới tới câu, tới từ.
    #    -> Sử dụng khi: Cần cân bằng giữa việc giữ lại cấu trúc đoạn văn và đảm bảo giới hạn độ dài.
    # 2. 5.1 Structural Chunking (Cắt theo cấu trúc): Dùng Regex tìm các điểm ngắt logic (\nChương..., \nĐiều...).
    #    -> Sử dụng khi: Làm việc với các văn bản có tính cấu trúc nghiêm ngặt (Luật pháp, Hợp đồng, Báo cáo tài chính). 
    #    -> Ưu điểm: Giữ được trọn vẹn 1 ý tưởng (1 Điều luật) trong 1 chunk, tránh bị mất ngữ nghĩa.
    # =================================================================
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=[
            r"\nChương\s+[IVXLCDM]+",    
            r"\nĐiều\s+\d+",             
            r"\n\d+\.\s",                
            r"\n[a-z]\)\s",              
            r"\n\n",                     
            r"\n",                       
            r"\s+",                      
            r""                          
    ],
        is_separator_regex=True          
    )
    chunks = splitter.split_documents(pages)
    print(f" Đã chia file {pdf_path} thành {len(chunks)} chunks.")
    print(f" Đang vector hóa và insert dữ liệu của {pdf_path} vào ChromaDB...")

    collection.upsert(
        ids=[f"{pdf_path}_chunk_{i}" for i, _ in enumerate(chunks)],
        documents=[c.page_content for c in chunks],
        metadatas=[{"source": pdf_path, "page": c.metadata.get("page", 0)}
                   for c in chunks],
    )
    print(f"Ingested {len(chunks)} chunks from {pdf_path}.")

# ─────────────────────────────────────────
# 2.5 BATCH PDF INGESTION HELPER (THÊM MỚI)
# ─────────────────────────────────────────
def ingest_folder(folder_path: str):
    """
    Quét và nạp toàn bộ file PDF trong một thư mục cùng lúc.
    
    =================================================================
    GHI CHÚ TỪ TÀI LIỆU VỀ CÁC CẤP ĐỘ NẠP NHIỀU FILE:
    
    - Cấp độ 1: Xử lý theo thư mục (Vòng lặp cơ bản):
      + Khi nào sử dụng: Khi chỉ có số lượng file ít, nạp test.
      + Cách hoạt động: Dùng vòng lặp for nạp tuần tự (xong file 1 mới tới file 2).
      + Hạn chế: Nạp 100 file sẽ mất hàng giờ, rất chậm.

    - Cấp độ 2: Xử lý Đa luồng / Bất đồng bộ (ĐANG ĐƯỢC ÁP DỤNG DƯỚI ĐÂY):
      + Khi nào sử dụng: KHUYÊN DÙNG CHO WORKSHOP. Khi cần nạp 10 - 100 file nhanh chóng trên máy tính cá nhân.
      + Cách hoạt động: Sử dụng thư viện concurrent.futures (Parallel Processing) để chạy nạp nhiều file cùng một lúc.

    - Cấp độ 3: Tối ưu Database (Bulk Upsert):
      + Khi nào sử dụng: Trong môi trường Doanh nghiệp (Enterprise), khi hệ thống mạng có nguy cơ bị quá tải Rate Limit API.
      + Cách hoạt động: Đọc và cắt chunk 100 file ở Local -> Gom toàn bộ thành 1 mảng khổng lồ (vd: 10,000 chunks) -> Gửi lệnh collection.upsert(...) đúng 1 lần duy nhất lên Chroma Cloud.
    =================================================================
    """
    if not os.path.exists(folder_path):
        print(f"Không tìm thấy thư mục: {folder_path}")
        return
        
    # Tìm tất cả các file .pdf trong thư mục
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"Không có file PDF nào trong thư mục {folder_path}")
        return
        
    print(f" Tìm thấy {len(pdf_files)} file PDF. Đang áp dụng Cấp độ 2 (Đa luồng/Parallel Processing) để nạp...")
    
    # Sử dụng ThreadPoolExecutor để chạy đa luồng (Cấp độ 2)
    # max_workers=5 nghĩa là nạp tối đa 5 file cùng một lúc để tránh nghẽn máy
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Tạo danh sách các task
        futures = [executor.submit(ingest_pdf, pdf_path) for pdf_path in pdf_files]
        
        # Chờ đến khi toàn bộ file được nạp xong
        concurrent.futures.wait(futures)
        
    print("\n ĐÃ HOÀN TẤT NẠP TOÀN BỘ FILE TRONG THƯ MỤC!")

# ─────────────────────────────────────────
# 3. THE RETRIEVAL TOOL
# ─────────────────────────────────────────
@tool
def retrieve_from_pdf(query: str, n_results: int = 4) -> str:
    """
    Tìm kiếm trong cơ sở dữ liệu PDF Luật Dân số Việt Nam và trả về các đoạn văn liên quan nhất.
    
    QUAN TRỌNG - QUY TẮC VỀ THAM SỐ 'query':
    - Tham số 'query' PHẢI là câu hỏi GỐC của người dùng bằng tiếng Việt.
    - KHÔNG dịch sang tiếng Anh.
    - KHÔNG sửa chính tả hoặc dấu thanh.
    - KHÔNG tóm tắt hoặc rút gọn.
    - KHÔNG thay đổi từ ngữ.
    - Sao chép NGUYÊN VĂN câu hỏi của người dùng.
    """
    print(f"\n[Tool] Retrieving for: '{query}'")

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    passages  = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not passages:
        return "Không tìm thấy đoạn văn liên quan trong cơ sở dữ liệu."

    formatted = []
    for i, (text, meta) in enumerate(zip(passages, metadatas), 1):
        source = meta.get("source", "unknown")
        page   = meta.get("page", "?")
        formatted.append(f"[Đoạn {i} — {source}, trang {page}]\n{text}")

    return "\n\n---\n\n".join(formatted)

# ─────────────────────────────────────────
# 4. LLM + GRAPH  
# ─────────────────────────────────────────
tools = [retrieve_from_pdf]

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """Bạn là trợ lý AI có quyền truy cập cơ sở dữ liệu PDF về Luật Dân số Việt Nam.

QUY TẮC BẮT BUỘC:
1. Khi người dùng hỏi về luật, điều khoản, chương, nguyên tắc, hoặc bất kỳ nội dung văn bản nào:
   PHẢI gọi công cụ retrieve_from_pdf TRƯỚC KHI trả lời.
2. Khi gọi retrieve_from_pdf:
   - Sao chép NGUYÊN VĂN câu hỏi của người dùng vào tham số 'query'.
   - TUYỆT ĐỐI KHÔNG dịch sang tiếng Anh.
   - TUYỆT ĐỐI KHÔNG sửa chính tả hoặc dấu thanh tiếng Việt.
   - TUYỆT ĐỐI KHÔNG tóm tắt hoặc đổi từ.
3. Sau khi nhận được các đoạn văn:
   - Trả lời dựa CHÍNH XÁC trên nội dung đoạn văn.
   - Nếu đoạn văn không đủ thông tin, nói rõ "Không tìm thấy thông tin này trong tài liệu."
   - Luôn ghi rõ số trang và nguồn (ví dụ: "Theo trang 5...").
4. Luôn trả lời bằng tiếng Việt.
5. Đối với lời chào hoặc câu hỏi không liên quan đến tài liệu: trả lời trực tiếp, không cần gọi công cụ.
"""

def agent_node(state: MessagesState) -> dict:
    print(f"[Agent] Thinking... ({len(state['messages'])} messages in context)")
    from langchain_core.messages import SystemMessage
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()

try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("Graph diagram saved to graph.png")
except Exception as e:
    print(graph.get_graph().draw_mermaid())


# ─────────────────────────────────────────
# 5. MAIN LOOP
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\nRAG Agent ready.")
    print("Commands:")
    print("  ingest <path/to/file.pdf>     — load a PDF into the knowledge base")
    print("  ingest_folder <path/to/folder>— BỔ SUNG: Nạp ĐA LUỒNG toàn bộ file PDF trong một thư mục (Level 2)")
    print("  quit                          — exit")
    print("  anything else                 — ask the agent\n")

    conversation_history = []   

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        # ── ingest command (1 file) ──
        if user_input.lower().startswith("ingest "):
            pdf_path = user_input[7:].strip()
            if os.path.exists(pdf_path):
                ingest_pdf(pdf_path)
            else:
                print(f"File not found: {pdf_path}")
            continue

        # ── BỔ SUNG: ingest_folder command (Nhiều file) ──
        elif user_input.lower().startswith("ingest_folder "):
            folder_path = user_input[14:].strip()
            ingest_folder(folder_path)
            continue

        if user_input.lower() == "quit":
            break

        # ── normal question ──
        conversation_history.append(HumanMessage(content=user_input))
        result = graph.invoke({"messages": conversation_history})
        conversation_history = result["messages"]
        final_answer = result["messages"][-1].content
        print(f"\nAgent: {final_answer}\n")