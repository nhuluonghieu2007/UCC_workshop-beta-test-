# query only
import os
from dotenv import load_dotenv
#from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ─────────────────────────────────────────
# 1. CONNECT TO YOUR CHROMA CLOUD COLLECTION
# ─────────────────────────────────────────
# This reuses exactly what you built in Demo 4.
# HttpClient connects to Chroma Cloud with your credentials.

client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CHROMA_API_KEY")
)
#
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
collection_name = "luat_dan_so_vietnam8"
try:
    collection = client.get_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef,
    )
except Exception as e:
    print(f"❌ Không tìm thấy collection '{collection_name}'.")
    print(f"   Lỗi: {e}")
    print(f"   Hãy chạy script ingest trước để tạo collection.")
    exit(1)

chunk_count = collection.count()
print(f"✅ Đã kết nối Collection: '{collection_name}' ({chunk_count} chunks)")

if chunk_count == 0:
    print("⚠️  Cảnh báo: Collection rỗng. Hãy ingest dữ liệu trước.")
    exit(1)
print(f" Đã tạo/kết nối thành công Collection: '{collection_name}'")
# ─────────────────────────────────────────
# 3. THE RETRIEVAL TOOL
# ─────────────────────────────────────────
# your ChromaDB query becomes a @tool the LLM can call.
#
# The docstring is critical — the LLM reads it to decide
# WHEN to call this tool. Be specific and honest about what
# the collection contains.

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
    
    Ví dụ đúng:
    - Người dùng: "điều 3 quy định nguyên tắc gì?"
    - query = "điều 3 quy định nguyên tắc gì?"
    
    Ví dụ SAI (KHÔNG được làm như thế này):
    - query = "nguyên tắc điều 3"  ← SAI vì đã rút gọn
    - query = "principles of article 3"  ← SAI vì đã dịch
    - query = "nguên tắc điên 3"  ← SAI vì sai chính tả
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
# 4. LLM + GRAPH  (same pattern as Demo 3)
# ─────────────────────────────────────────

tools = [retrieve_from_pdf]

#llm = ChatOllama(model="qwen2.5:7b", temperature=0)
llm = ChatGroq(model="qwen3-32b", temperature=0)
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
    # Prepend the system prompt on every call
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
# Save a PNG of the graph structure
try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("Graph diagram saved to graph.png")
except Exception as e:
    # Falls back to text representation if rendering fails
    print(graph.get_graph().draw_mermaid())


# ─────────────────────────────────────────
# 5. MAIN LOOP (query-only)
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\n🤖 RAG Agent sẵn sàng — Luật Dân số Việt Nam")
    print("   Gõ câu hỏi của bạn, hoặc 'quit' để thoát.\n")

    conversation_history = []   # keeps multi-turn context

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "thoát"):
            print("Tạm biệt!")
            break

        # ── ask the agent ──
        conversation_history.append(HumanMessage(content=user_input))

        try:
            result = graph.invoke({"messages": conversation_history})
            conversation_history = result["messages"]
            final_answer = result["messages"][-1].content
            print(f"\nAgent: {final_answer}\n")
        except Exception as e:
            print(f"\n❌ Lỗi: {e}\n")
            # Roll back the failed user message so history stays clean
            conversation_history.pop()