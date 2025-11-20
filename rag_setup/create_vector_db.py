import os
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# 1. Path to your pre-processed MTG card data CSV
# IMPORTANT: Adjust this path based on where you put your file.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE_PATH = BASE_DIR / "data" / "commander_legal_cards.csv"

# 2. Vector DB settings
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "mtg_commander_cards"

# 3. Ollama Embedding Model (Must be pulled locally first)
EMBEDDING_MODEL = "nomic-embed-text"

def create_db_from_csv():
    """
    Loads MTG card data from CSV, creates embeddings using Ollama, 
    and stores the documents in a persistent ChromaDB instance.
    """
    print(f"--- Starting Vector Database Creation ---")
    
    # 1. Load Documents
    print(f"Loading data from: {DATA_FILE_PATH}")
    
    # The CSVLoader turns each row into a LangChain Document.
    # It intelligently combines all columns into the 'page_content' text block.
    try:
        loader = CSVLoader(
            file_path=str(DATA_FILE_PATH), 
            # We specify the column that will serve as the unique source ID (the card name)
            source_column="Name" 
        )
        documents = loader.load()
        print(f"✅ Documents loaded. Total cards: {len(documents)}")
    except FileNotFoundError:
        print(f"❌ ERROR: Data file not found at {DATA_FILE_PATH}. Please copy your CSV.")
        return

    # 2. Define the Embedding Function
    # We use OllamaEmbeddings to generate vector representations for the text data.
    try:
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL, 
            base_url="http://localhost:11434"
        )
        print(f"✅ Embedding model '{EMBEDDING_MODEL}' initialized via Ollama.")
    except Exception as e:
        print(f"❌ ERROR connecting to Ollama embeddings: {e}")
        print("Ensure 'ollama serve' is running and the model is pulled.")
        return

    # 3. Create and Persist the Vector Store
    print(f"Indexing data into ChromaDB at: {CHROMA_DB_PATH}")
    
    # Chroma.from_documents creates the vectors and stores them on disk (persists them).
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_PATH) # Saves the DB to a folder
    )

    vectorstore.persist()
    print("--- Indexing Complete ---")
    print(f"Vector Database saved successfully with {vectorstore._collection.count()} embeddings.")
    
    # Optional: Test a simple semantic query
    query = "cards that create lots of creatures fast"
    print(f"\n--- Testing Retrieval for: '{query}' ---")
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    
    for doc in retrieved_docs:
        print(f"Score: {doc.metadata.get('hnsw_score', 'N/A')}")
        print(f"Card: {doc.metadata.get('source')}")
        
    print("---------------------------------------")


if __name__ == "__main__":
    create_db_from_csv()