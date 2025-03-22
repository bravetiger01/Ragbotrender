# Main application with RAG pipeline
import os
import json
import numpy as np
import requests
import flask
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from main import config
from main.processor import DataProcessor
from main.retriever import SimpleRetriever
import nltk
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for all routes


# Global variables for the RAG system
processor = None
retriever = None
chunks = None
embeddings = None
embedding_model = None

def initialize_system():
    """Initialize the RAG system by loading models and processing data."""
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        print("Downloading NLTK punkt tokenizer data...")
        nltk.download('punkt_tab')
        nltk.download('punkt')
    
    global retriever, processor, embedding_model
    
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    print(f"Loaded embedding model: {config.EMBEDDING_MODEL}")
    
    processor = DataProcessor()
    
    if (os.path.exists(config.CHUNKS_PATH) and 
        os.path.exists(config.VECTORS_PATH)):
        
        print(f"Loading preprocessed data...")
        with open(config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        embeddings = np.load(config.VECTORS_PATH)
        print(f"Loaded {len(chunks)} chunks and their embeddings")
    
    else:
        if not os.path.exists(config.RAW_DATA_PATH):
            raise FileNotFoundError(f"Raw data file not found at {config.RAW_DATA_PATH}")
        
        print(f"Processing raw data...")
        raw_data = processor.load_json_data(config.RAW_DATA_PATH)
        chunks = processor.hierarchical_chunking(raw_data)
        embeddings = processor.generate_embeddings(chunks)
        
        processor.save_json_data(chunks, config.CHUNKS_PATH)
        np.save(config.VECTORS_PATH, embeddings)
    
    retriever = SimpleRetriever(chunks, embeddings, processor)
    print("RAG system initialized successfully!")

def query_gemini(prompt):
    """Query the Google Gemini API"""
    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024
        }
    }
    
    try:
        response = requests.post(
            f"{config.GEMINI_API_URL}?key={config.GEMINI_API_KEY}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Error ({response.status_code}): {response.text}"
    
    except Exception as e:
        return f"Error: {str(e)}"

def process_query(query):
    """Process a query through the RAG pipeline"""
    query_embedding = embedding_model.encode(query)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    retrieved_chunks = retriever.hybrid_search(
        query_text=query,
        query_embedding=query_embedding,
        top_k=config.TOP_K
    )
    
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_part = f"[Document {i+1}] {chunk['metadata']['title']}\n"
        if chunk['metadata'].get('url'):
            context_part += f"Source: {chunk['metadata']['url']}\n"
        context_part += f"Content: {chunk['text']}\n"
        context_parts.append(context_part)
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful assistant. Answer the question based on the provided context.
If the answer isn't in the context, respond "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer in clear, concise English:"""
    
    response = query_gemini(prompt)
    
    return {
        "query": query,
        "answer": "Helllo world",
        "sources": [chunk["metadata"] for chunk in retrieved_chunks]
    }

@app.route('/')
def index():
    return "RAG System with Google Gemini is running!"

@app.route('/ask', methods=['POST', 'GET'])
def ask():
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        result = process_query(query)

        print("Response:", result)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"abc + {str(e)}"}), 500  # Ensure errors return JSON

@app.route('/chat')
def chat_interface():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Chat Interface</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
            #chatbox { height: 400px; border: 1px solid #ddd; padding: 10px; overflow-y: auto; margin-bottom: 10px; }
            #input { width: 80%; padding: 10px; }
            button { padding: 10px 20px; }
        </style>
    </head>
    <body>
        <h1>RAG Chat Interface (Google Gemini)</h1>
        <div id="chatbox"></div>
        <input id="input" type="text" placeholder="Ask a question...">
        <button onclick="sendMessage()">Send</button>

        <script>
            function sendMessage() {
                const input = document.getElementById('input');
                const query = input.value;
                if (!query) return;
                
                addMessage('User: ' + query);
                input.value = '';
                
                fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                })
                .then(response => response.json())
                .then(data => {
                    addMessage('Bot: ' + data.answer);
                    
                    if (data.sources && data.sources.length > 0) {
                        let sourceText = 'Sources: ';
                        data.sources.forEach((source, index) => {
                            if (source.title) {
                                sourceText += source.title;
                                if (index < data.sources.length - 1) sourceText += ', ';
                            }
                        });
                        addMessage(sourceText);
                    }
                })
                .catch(error => {
                    addMessage('Error: abc2 ' + error);
                });
            }
            
            function addMessage(message) {
                const chatbox = document.getElementById('chatbox');
                const messageElement = document.createElement('div');
                messageElement.textContent = message;
                chatbox.appendChild(messageElement);
                chatbox.scrollTop = chatbox.scrollHeight;
            }
            
            document.getElementById('input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    """

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "API is working"})

# Global initialization - outside of if __name__ == '__main__'
try:
    print("System initialized successfully at application startup!")
except Exception as e:
    print(f"⚠️ System initialization failed: {str(e)}")

if __name__ == '__main__':
    initialize_system()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)