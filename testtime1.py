import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# -------------------------------------------------------------------
# BƯỚC 1: KẾT NỐI CLOUD CHROMADB
# -------------------------------------------------------------------
# Load biến môi trường
load_dotenv()

client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CHROMA_API_KEY")
)
# -------------------------------------------------------------------
# BƯỚC 2: ĐỌC FILE PDF
# -------------------------------------------------------------------
pdf_path = "luatdanso.pdf" # Đảm bảo file này nằm cùng thư mục với code
print(f"⏳ [Bước 2] Đang đọc file PDF: {pdf_path}...")

# PyPDFLoader của LangChain giúp parse text từ PDF cực kỳ ổn định
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Đã đọc xong! PDF có tổng cộng {len(documents)} trang.")

# -------------------------------------------------------------------
# BƯỚC 3: CHUNKING FILE PDF
# -------------------------------------------------------------------
print("⏳ [Bước 3] Đang tiến hành Chunking text...")

# Sử dụng RecursiveCharacterTextSplitter để chia nhỏ văn bản.
# chunk_size: Số lượng ký tự tối đa trong 1 chunk (khoảng 1000 là đẹp cho mô hình embedding)
# chunk_overlap: Số ký tự chồng lấn giữa các chunk để không bị mất ngữ cảnh ở đoạn cắt
#
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=[
        r"\nChương\s+[IVXLCDM]+",    # Đã sửa \n\n thành \n để dễ bắt hơn
        r"\nĐiều\s+\d+",             # Đã sửa \n\n thành \n
        r"\n\d+\.\s",                
        r"\n[a-z]\)\s",              
        r"\n\n",                     
        r"\n",                       
        r"\s+",                      
        r""                          
    ],
    is_separator_regex=True          # KÍCH HOẠT CHẾ ĐỘ REGEX
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Đã chia file PDF thành {len(chunks)} chunks.")

# -------------------------------------------------------------------
# BƯỚC 4: TẠO hàm embedding và DATABASE (COLLECTION) TRONG CHROMADB
# -------------------------------------------------------------------
print("⏳ [Bước 4] Đang khởi tạo Collection (Database)...")

# Khởi tạo hàm Embedding. Tại đây ta sẽ sử dụng mode paraphrase-multilingual-MiniLM-L12-v2
# Hàm này sẽ chuyển hóa text (chunks) thành các vector số học (dense vectors).
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# Tạo collection. Nếu đã có thì lấy ra dùng tiếp (get_or_create)
collection_name = "luat_dan_so_vietnam"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=sentence_transformer_ef
)
print(f"✅ Đã tạo/kết nối thành công Collection: '{collection_name}'")

# -------------------------------------------------------------------
# BƯỚC 5: NỐI KNOWLEDGE (INSERT CHUNKS) VÀO DATABASE
# -------------------------------------------------------------------
print("⏳ [Bước 5] Đang vector hóa và insert dữ liệu vào ChromaDB...")

# Tách dữ liệu từ LangChain format sang format mà ChromaDB hiểu
texts = [chunk.page_content for chunk in chunks]
# Giữ lại metadata (ví dụ: source file, số trang) để sau này truy xuất dễ hơn
metadatas = [chunk.metadata for chunk in chunks] 
# Tạo ID độc nhất cho mỗi chunk
ids = [f"chunk_{i}" for i in range(len(chunks))]

# Đẩy dữ liệu vào Database. 
# Bước này Chroma sẽ tự động gọi hàm embedding_function ở trên để biến 'texts' thành vectors.
collection.upsert(
    documents=texts,
    metadatas=metadatas,
    ids=ids
)
print("✅ Hoàn tất! Knowledge đã được lưu vào Database.")

# --- Test thử query nhẹ nhàng để xem DB có hoạt động không ---
print("\n🔍 Đang test truy xuất dữ liệu (Similarity Search)...")
results = collection.query(
    query_texts=["Điều 3 trong bộ luật dân số quy định những nguyên tắc gì?"],
    n_results=1 # Lấy ra 1 chunk có độ tương đồng cao nhất
)

print(f"📄 Kết quả tìm thấy (Top 1): {results['documents'][0][0]}")