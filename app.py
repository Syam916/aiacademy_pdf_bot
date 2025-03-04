from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.openapi.shared.exceptions import PineconeApiException
from dotenv import load_dotenv
from typing import Optional
import uvicorn
from pydantic import BaseModel

# Load environment variables
load_dotenv()

app = FastAPI(title="Chemistry School Teacher")

# Initialize global variables
vector_store = None
qa_chain = None
pdf_processed = False

# Set your API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

# Define request model
class QuestionRequest(BaseModel):
    question: str
    pdf_file: Optional[UploadFile] = None

def initialize_pinecone():
    """Initialize Pinecone and create index if it doesn't exist"""
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
    return pc, index_name

def process_pdf_document(file_path, index_name):
    """Process the PDF and store it in Pinecone"""
    global vector_store, qa_chain, pdf_processed
    
    # Process the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and store in Pinecone
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    # Store embeddings in Pinecone
    vector_store = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    # Create the QA chain with custom prompt
    template = """
    You are a friendly and knowledgeable chemistry school teacher. 
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Present your answer in a storytelling format with real-life examples and some humor.
    Make sure your explanation is structured with clear sections and is easy to understand for a school student.
    
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
    
    pdf_processed = True
    return True

def setup_existing_vector_store(index_name):
    """Set up vector store from existing Pinecone index without reprocessing PDF"""
    global vector_store, qa_chain, pdf_processed
    
    if pdf_processed:
        return True
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    # Connect to existing Pinecone index
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    # Create the QA chain with custom prompt
    template = """
    You are a friendly and knowledgeable chemistry school teacher. 
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Present your answer in a storytelling format with real-life examples and some humor.
    Make sure your explanation is structured with clear sections and is easy to understand for a school student.
    
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
    
    pdf_processed = True
    return True

def get_answer(question):
    """Get answer for a question using the QA chain"""
    global qa_chain
    
    if not qa_chain:
        raise HTTPException(status_code=400, detail="Please upload a chemistry textbook first!")
    
    # Get answer from the QA chain
    answer = qa_chain.run(question)
    return {"answer": answer}

# Initialize Pinecone on startup
pc, index_name = initialize_pinecone()

@app.post("/process_and_answer")
async def process_and_answer(question: str = Form(...), file: Optional[UploadFile] = None):
    global pdf_processed
    
    # If we have PDF file, process it
    if file:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")
        
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp:
            temp_path = temp.name
            contents = await file.read()
            with open(temp_path, 'wb') as f:
                f.write(contents)
        
        try:
            # Process the PDF
            process_pdf_document(temp_path, index_name)
            
            # Clean up the temp file
            os.unlink(temp_path)
            
        except Exception as e:
            # Clean up the temp file in case of error
            os.unlink(temp_path)
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # If no PDF but we need to initialize from existing index
        if not pdf_processed:
            try:
                setup_existing_vector_store(index_name)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error initializing from existing index: {str(e)}")
    
    # Now answer the question
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    
    try:
        return get_answer(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8002)