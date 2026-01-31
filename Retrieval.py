# retrieval.py - Query EBA guide with RAG
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME", "eba-credit-risk")
index = pc.Index(index_name)

# Initialize embeddings + vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize Groq
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def query_eba_guide(question: str, top_k: int = 3):
    """Query EBA guide and return answer with sources"""
    
    # Retrieve relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    results = retriever.invoke(question)
    
    # Build context
    context = "\n\n".join([f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}" 
                           for doc in results])
    
    # Create prompt
    prompt = f"""You are an expert in credit risk modeling and EBA regulatory guidelines.

Based on the EBA Methodological Guide context below, answer the question accurately.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, technical answer based ONLY on the context. Cite page numbers when relevant.

ANSWER:"""
    
    # Get response from Groq
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert in credit risk and EBA guidelines. Answer based only on provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1024,
    )
    
    answer = chat_completion.choices[0].message.content
    sources = [{"page": doc.metadata.get("page", "N/A"), "text": doc.page_content[:200]} 
               for doc in results]
    
    return {"answer": answer, "sources": sources}


# Test query
if __name__ == "__main__":
    result = query_eba_guide("What are the main asset quality indicators (AQT)?")
    print("\nANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for i, src in enumerate(result["sources"], 1):
        print(f"{i}. Page {src['page']}: {src['text']}...")