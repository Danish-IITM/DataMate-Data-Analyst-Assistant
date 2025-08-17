# app.py (No changes needed)

import os
import uuid
import shutil
import json
from flask import Flask, request, jsonify, render_template
from agent import DataAnalystAgent
import traceback
from dotenv import load_dotenv
from google.api_core import exceptions as google_exceptions

load_dotenv()
app = Flask(__name__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not os.path.exists("temp"):
    os.makedirs("temp")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/', methods=['POST'])
def handle_analysis_request():
    print("Received a new request to /api/")
    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY is not set.")
        return jsonify({"error": "API key is not configured on the server."}), 500
    if 'questions.txt' not in request.files:
        return jsonify({"error": "'questions.txt' is a required file part."}), 400

    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp", request_id)
    os.makedirs(temp_dir)
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        uploaded_files = []
        for filename, file_storage in request.files.items():
            sanitized_filename = os.path.basename(filename)
            file_path = os.path.join(temp_dir, sanitized_filename)
            file_storage.save(file_path)
            if sanitized_filename != "questions.txt":
                uploaded_files.append(sanitized_filename)
        
        question_path = os.path.join(temp_dir, "questions.txt")
        with open(question_path, 'r', encoding='utf-8') as f:
            question_content = f.read()
            
        agent = DataAnalystAgent(api_key=GOOGLE_API_KEY, work_dir=temp_dir)
        result_json_str = agent.run(question=question_content, files=uploaded_files)
        response_data = json.loads(result_json_str)
        
        print("Successfully processed request. Sending JSON response.")
        return jsonify(response_data)

    except (google_exceptions.RetryError, google_exceptions.DeadlineExceeded) as e:
        print(f"API Timeout/RetryError for request {request_id}: {e}")
        return jsonify({
            "error": "The analysis service API is currently overloaded or the request timed out.", 
            "details": str(e)
        }), 503

    except Exception as e:
        print(f"An unexpected internal error occurred for request {request_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500
    
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)