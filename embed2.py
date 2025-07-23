import openai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
import uuid
from openai import OpenAI
import keys
import fitz  # PyMuPDF

# Load keys
QDRANT_API_KEY = keys.QDRANT_API_KEY
QDRANT_URL = keys.QDRANT_URL
OPENAI_API_KEY = keys.OPENAI_API_KEY

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Qdrant client
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "Stratford_Nissan_Knowledge_Base"

# Step 1: Recreate Qdrant collection
# qdrant.recreate_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
# )

qdrant.get_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Step 2: Read and split text from PDF
def extract_text_from_pdf(pdf_path: str, max_chunk_chars: int = 1000):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Split text into chunks
    chunks = []
    while full_text:
        chunk = full_text[:max_chunk_chars]
        chunks.append(chunk.strip())
        full_text = full_text[max_chunk_chars:]
    return chunks

# PDF file path
pdf_path = "Stratford_Nissan_Knowledge_Base_Final.pdf"  # <-- Replace with your PDF file

# Extracted document chunks
documents = extract_text_from_pdf(pdf_path)

# Step 3: Embed and upload to Qdrant
batch = []
for doc in tqdm(documents):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    embedding = response.data[0].embedding

    doc_id = str(uuid.uuid4())

    point = PointStruct(
        id=doc_id,
        vector=embedding,
        payload={"text": doc}
    )

    batch.append(point)

# Upload all points
qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=batch
)

print(f"✅ Successfully uploaded {len(batch)} chunks from PDF to Qdrant.")
