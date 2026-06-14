import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# 1. BẮT BUỘC: Kết nối cloud
# Load biến môi trường
load_dotenv()

client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CHROMA_API_KEY")
)

print("Kết nối Chroma Cloud thành công!")

# 2. BẮT BUỘC: Khai báo lại model embedding ĐÃ DÙNG LÚC TẠO
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# 3. KẾT NỐI VÀ QUERY
collection = client.get_collection( # Dùng get_collection thay vì get_or_create
    name="luat_dan_so_vietnam",
    embedding_function=sentence_transformer_ef
)

results = collection.query(
    query_texts=["Điều 3 quy định nguyên tắc gì?"],
    n_results=1
)
print(results['documents'][0][0])