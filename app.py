from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from pinecone.core.openapi.shared.exceptions import PineconeApiException

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize global variables
vector_store = None
qa_chain = None

# Set your API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

# Initialize Pinecone with the new method
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'chemistry-grade7'
try:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
except PineconeApiException as e:
    # Check if the error message contains "409"
    if "409" in str(e):
        print("Index already exists. Skipping creation.")
    else:
        raise

    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_store, qa_chain
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
            file.save(temp.name)
            temp_path = temp.name
        
        try:
            # Process the PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # Split the documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            # Create embeddings and store in Pinecone
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            
            # Store embeddings in Pinecone using the updated method
            vector_store = PineconeVectorStore.from_documents(
                documents=texts,
                embedding=embeddings,
                index_name=index_name,
                pinecone_api_key=PINECONE_API_KEY
            )
            
            # Create the QA chain with custom prompt
            template = """
            You are a friendly and knowledgeable 7th-grade chemistry teacher. 
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Present your answer in a storytelling format with real-life examples and some humor.
            Make sure your explanation is structured with clear sections and is easy to understand for a 7th-grade student.
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
            
            PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                api_key=OPENAI_API_KEY
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            # Clean up the temp file
            os.unlink(temp_path)
            
            return jsonify({"message": "PDF processed successfully!"}), 200
            
        except Exception as e:
            # Clean up the temp file in case of error
            os.unlink(temp_path)
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global qa_chain
    
    if not qa_chain:
        return jsonify({"error": "Please upload a chemistry textbook first!"}), 400
    
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        # Get answer from the QA chain
        answer = qa_chain.run(question)
        return jsonify({"answer": answer}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 