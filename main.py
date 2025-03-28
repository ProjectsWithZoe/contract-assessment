import pdfplumber
import tiktoken
import numpy as np
import openai
import faiss
from dotenv import load_dotenv
import os

load_dotenv()

# Create a single client instance to use throughout the code
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#open pdf file that needs assessing
def openFile(pdf_file):
    text = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return '\n'.join(text)

#tokenizer to create chunks
def chunk_text(text, max_tokens=512):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(text)

    chunks = []
    for i in range(0,len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def generate_embeddings(chunks):
    response = client.embeddings.create(  # Use the global client
        input=chunks,
        model="text-embedding-ada-002"
    )
    embeddings = np.array([r.embedding for r in response.data])
    return embeddings

def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index    

def retrieve_similar_chunks(query, index, chunks, top_k=5):
    query_embedding = generate_embeddings([query])
    distances, indices = index.search(query_embedding, top_k)
    similar_chunks = [chunks[i] for i in indices[0]]
    return similar_chunks

def query_chatgpt(query, similar_chunks):
    context = "\n".join(similar_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = client.completions.create(  # Use the global client
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

def main():
    pdf_file = 'CTAC.pdf'
    pdf_text = openFile(pdf_file)
    chunks = chunk_text(pdf_text)
    embeddings = generate_embeddings(chunks)
    index = build_faiss_index(embeddings)
    query = "When was this file created?"
    similar_chunks = retrieve_similar_chunks(query, index, chunks)
    answer = query_chatgpt(query, similar_chunks)
    print(answer)
if __name__ == "__main__":
    main()



