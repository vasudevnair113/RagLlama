from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from sentence_transformers import SentenceTransformer
import os
import PyPDF2
import streamlit as st

st.rerun()

# Initialize Groq client
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])

# Initialize Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index_name = "pdf-index"

# Check if the index exists, if not create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def index_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    vectors = []

    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        vectors.append((str(i), embedding, {"text": chunk}))
        progress_bar.progress((i + 1) / len(chunks))
    
    # Batch upsert
    batch_size = 100  # Adjust this value based on your needs
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    progress_bar.empty()

def query_pdf(query):
    query_embedding = model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    context = " ".join([result.metadata['text'] for result in results.matches])
    
    prompt = f"""Context: {context}

Question: {query}

Please provide a detailed answer based on the given context."""

    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=500
    )
    
    return response.choices[0].message.content

st.title("Rag chatbot using Meta Llama, Groq, and Pinecone")
st.subheader("Created by Vasudev Nair")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write("Indexing PDF...")
    index_pdf(uploaded_file)
    st.write("PDF indexed successfully!")

    query = st.text_input("Enter your query about the PDF:")
    if query:
        st.write("Generating answer...")
        answer = query_pdf(query)
        st.write("Answer:")
        st.write(answer)
