import pdfplumber
import tiktoken
import numpy as np
import openai
import faiss
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    #print("Chunks", chunks)
    return chunks

def generate_embeddings(chunks):
    # Use TF-IDF instead of OpenAI embeddings
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks).toarray()
    return embeddings, vectorizer

def retrieve_similar_chunks(query, embeddings, chunks, vectorizer, top_k=5):
    # Transform query using same vectorizer
    query_vector = vectorizer.transform([query]).toarray()
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, embeddings)[0]
    
    # Get top k similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    similar_chunks = [chunks[i] for i in top_indices]
    return similar_chunks

def query_chatgpt(query, similar_chunks):
    context = "\n".join(similar_chunks)
    # Add explicit instructions to prevent hallucinations
    prompt = f"""Based STRICTLY on the provided context from the PDF document, please answer the question.
If the information cannot be found in the context, respond with 'Information not found in document.'
Do not make assumptions or use external knowledge.

Context: {context}

Question: {query}

If the question is about a date, please return it in the format:
{{
    "date": "YYYY-MM-DD",
    "confidence": "HIGH/MEDIUM/LOW",
    "source_text": "exact text from context that contains this date"
}}

If the question is not about a date, provide a direct answer using only information from the context."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a precise document analyzer. Only use information from the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3  # Lower temperature for more focused responses
    )
    return response.choices[0].message.content.strip()

def main():
    pdf_file = 'CTAC.pdf'
    pdf_text = openFile(pdf_file)
    chunks = chunk_text(pdf_text)
    
    # Generate TF-IDF embeddings
    embeddings, vectorizer = generate_embeddings(chunks)
    
    query = "Who isMichael J. McGinn?"
    similar_chunks = retrieve_similar_chunks(query, embeddings, chunks, vectorizer)
    answer = query_chatgpt(query, similar_chunks)
    print(answer)

if __name__ == "__main__":
    main()



