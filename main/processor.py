# Data processing utilities
import json
import os
import re
import nltk
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
from main import config

# Download NLTK resources
nltk.download('punkt', quiet=True)

class DataProcessor:
    def __init__(self):
        """Initialize the data processor"""
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        print(f"Loaded embedding model: all-mpnet-base-v2")
    
    def load_json_data(self, file_path):
        """Load JSON data from file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Ensure all items in the list are dictionaries
            data = [json.loads(doc) if isinstance(doc, str) else doc for doc in data]
        
        print(f"Loaded {len(data)} documents from {file_path}")
        return data
    
    def save_json_data(self, data, file_path):
        """Save data to JSON file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved data to {file_path}")
    
    def hierarchical_chunking(self, documents):
        """Process documents with hierarchical chunking"""
        chunks = []
        
        for doc_id, doc in enumerate(tqdm(documents, desc="Processing documents")):
            # Extract document content
            content = doc.get("content", "")
            title = doc.get("title", "")
            url = doc.get("url", "")
            
            if not content:
                continue
            
            # Clean HTML if present
            content = self._clean_html(content)
            
            # First split into larger parent chunks
            parent_chunks = self._split_text(content, config.PARENT_CHUNK_SIZE, config.PARENT_CHUNK_OVERLAP)
            
            for parent_id, parent_chunk in enumerate(parent_chunks):
                # Split parent chunk into smaller chunks
                child_chunks = self._split_text(parent_chunk, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                
                for child_id, child_chunk in enumerate(child_chunks):
                    # Add chunk with metadata
                    chunks.append({
                        "text": child_chunk,
                        "metadata": {
                            "doc_id": doc_id,
                            "title": title,
                            "url": url,
                            "parent_id": f"parent_{doc_id}_{parent_id}",
                            "child_id": f"child_{doc_id}_{parent_id}_{child_id}",
                            "parent_summary": parent_chunk[:100] + "..." if len(parent_chunk) > 100 else parent_chunk
                        }
                    })
        
        return chunks
    
    def _clean_html(self, html_content):
        """Clean HTML content - basic version"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        # Fix whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _split_text(self, text, chunk_size, chunk_overlap):
        """Split text into chunks by sentences"""
        if not text:
            return []
            
        # Get sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds the chunk size and we already have content,
            # then store the current chunk and start a new one
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Keep some sentences for overlap
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def generate_embeddings(self, chunks):
        """Generate embeddings for all text chunks"""
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batches to save memory
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts)
            embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized_embeddings = embeddings_array / norms
        
        return normalized_embeddings
    
    def prepare_bm25_corpus(self, chunks):
        """Prepare corpus for BM25 search"""
        tokenized_corpus = []
        
        for chunk in tqdm(chunks, desc="Preparing BM25 corpus"):
            # Tokenize and lowercase
            tokens = word_tokenize(chunk["text"].lower())
            tokenized_corpus.append(tokens)
        
        return tokenized_corpus