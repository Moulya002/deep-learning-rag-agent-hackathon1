from pathlib import Path

from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager

def main():
    corpus_dir = Path("data/corpus")

    chunker = DocumentChunker()
    store = VectorStoreManager()

    print("Loading corpus files...")

    all_chunks = []

    for file_path in corpus_dir.iterdir():
        if file_path.suffix.lower() in [".md", ".pdf"]:
            print(f"Chunking: {file_path.name}")
            chunks = chunker.chunk_file(file_path)
            all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")

    result = store.ingest(all_chunks)

    print("Ingestion result:")
    print(result)


if __name__ == "__main__":
    main()