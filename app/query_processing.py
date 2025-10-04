from langchain.docstore.document import Document
import os
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from groq import Groq
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from langchain_cohere.embeddings import CohereEmbeddings


class Preprocessing:
    @staticmethod
    def process_all_pdfs(pdf_directory):
        """Process all PDF files in a directory"""
        all_documents = []
        pdf_dir = Path(pdf_directory)

        # Find all PDF files recursively
        pdf_files = list(pdf_dir.glob("**/*.pdf"))

        print(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()

                # Add source information to metadata
                for doc in documents:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_type'] = 'pdf'

                all_documents.extend(documents)
                print(f"  ✓ Loaded {len(documents)} pages")

            except Exception as e:
                print(f"  ✗ Error: {e}")

        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents

    @staticmethod
    def split_documents(documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into smaller chunks for better RAG performance"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

        # Show example of a chunk
        if split_docs:
            print(f"\nExample chunk:")
            print(f"Content: {split_docs[0].page_content[:200]}...")
            print(f"Metadata: {split_docs[0].metadata}")

        return split_docs


class EmbeddingManager:
  
    

    def __init__(self,model_name:str = "embed-multilingual-v3.0"):

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):

        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path)

        print(f"Loading embedding model: {self.model_name}")
        self.model = CohereEmbeddings(
            model=self.model_name,
            cohere_api_key = os.getenv("COHERE_API_KEY") # better to use env var!
        )
        print("Model loaded successfully.")
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:

        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.embed_documents(texts)
        embeddings = np.array(embeddings)
        print("embedding done")
        print(type(embeddings))
        print(embeddings.shape[1])
                # print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store, automatically handling dimension mismatches."""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "vector_store"):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_dim = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            existing_collections = [c.name for c in self.client.list_collections()]
            if self.collection_name in existing_collections:
                self.collection = self.client.get_collection(self.collection_name)
                print(f"Found existing collection: {self.collection_name}")
            else:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "PDF document embeddings for RAG"}
                )
                print(f"Created new collection: {self.collection_name}")

            print(f"Vector store initialized. Documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.

        Args:
            documents: List of LangChain Document objects
            embeddings: Corresponding embeddings for the documents (np.ndarray)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a numpy.ndarray")

        new_dim = embeddings.shape[1]

        # If collection has existing documents, check dimension
        if self.collection.count() > 0:
            try:
                first_emb = self.collection.get(ids=[self.collection.get()[0]['id']])['embeddings'][0]
                self.embedding_dim = len(first_emb)
                if self.embedding_dim != new_dim:
                    print(f"Dimension mismatch: collection has {self.embedding_dim}, got {new_dim}. Recreating collection...")
                    self.client.delete_collection(self.collection_name)
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "PDF document embeddings for RAG"}
                    )
                    self.embedding_dim = new_dim
                    print(f"New collection created with embedding dimension {new_dim}")
            except Exception:
                self.embedding_dim = new_dim
        else:
            # Empty collection → set embedding_dim
            self.embedding_dim = new_dim

        print(f"Adding {len(documents)} documents with dimension {new_dim} to vector store...")

        ids, metadatas, documents_text, embeddings_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise




class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


class RAG:
    @staticmethod
    def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
        results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
        if not results:
            return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}

        context = "\n\n".join([doc['content'] for doc in results])
        sources = [{
            'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
            'page_label': doc['metadata'].get('page_label', 'unknown'),
            'score': doc['similarity_score'],
            'preview': doc['content'][:300] + '...'
        } for doc in results]

        confidence = max([doc['similarity_score'] for doc in results])

        prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
        response = llm.invoke([prompt])

        output = {
            'answer': response.content,
            'sources': sources,
            'confidence': confidence
        }

        if return_context:
            output['context'] = context
        return output

