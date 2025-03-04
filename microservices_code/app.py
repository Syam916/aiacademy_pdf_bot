from flask import Flask, render_template, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Service URL
SERVICE_URL = "http://localhost:8002"  # Make sure this matches your FastAPI service port

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Check if there's a file in the request
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        
        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400
        
        try:
            # Send both the question and PDF to the service
            files = {'file': (file.filename, file, 'application/pdf')}
            data = {'question': question}
            response = requests.post(f"{SERVICE_URL}/process_and_answer", files=files, data=data)
            
            if response.status_code == 200:
                return jsonify(response.json()), 200
            else:
                return jsonify(response.json()), response.status_code
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # No file, just send the question
        try:
            data = {'question': question}
            response = requests.post(f"{SERVICE_URL}/process_and_answer", data=data)
            
            if response.status_code == 200:
                return jsonify(response.json()), 200
            else:
                return jsonify(response.json()), response.status_code
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)