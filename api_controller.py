# api_controller.py

import os
import json
from flask import Flask, request, jsonify, send_file
from pathlib import Path
from loguru import logger
import io
import mimetypes
from dotenv import load_dotenv
import sys

# Import Vercel Blob SDK's put function
from vercel_blob import put # pip install vercel-blob-py

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
# Assuming api_controller.py is in the project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.workflows.single_prompt_workflow import SinglePromptWorkflow
from config.settings import EXAMS_DIR, VERCEL_BLOB_READ_WRITE_TOKEN, VERCEL_BLOB_BASE_URL # Import new settings

app = Flask(__name__)

# Configure logging for the API
logger.remove() # Remove default console handler
logger.add(sys.stderr, level="INFO") # Add a console handler for API logs
# Ensure the logs directory exists for the API specific log file
api_logs_dir = EXAMS_DIR.parent / "logs"
api_logs_dir.mkdir(parents=True, exist_ok=True)
logger.add(f"{api_logs_dir}/api_requests.log", rotation="10 MB", level="INFO") # API specific log file

@app.route('/generate-exam', methods=['POST'])
async def generate_exam_endpoint():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        topic = data.get('topic')
        pdf_blob_urls = data.get('pdf_blob_urls') 
        requirements_file = data.get('requirements_file') # Optional, for custom requirements

        if not topic:
            return jsonify({"error": "Topic is required"}), 400
        if not pdf_blob_urls:
            return jsonify({"error": "PDF Blob URLs are required"}), 400
        
        logger.info(f"API Request: Received request to generate exam for topic '{topic}' with {len(pdf_blob_urls)} PDFs from Vercel Blob.")

        workflow = SinglePromptWorkflow()
        
        # Execute the workflow, passing the Vercel Blob URLs
        workflow_result = await workflow.execute_full_workflow(
            topic=topic,
            requirements_file=requirements_file,
            pdf_blob_urls=pdf_blob_urls
        )

        if workflow_result["workflow_metadata"]["success"]:
            generated_files_local_paths = workflow_result.get("output_files", [])
            
            uploaded_file_urls = []
            if not VERCEL_BLOB_READ_WRITE_TOKEN:
                logger.error("❌ VERCEL_BLOB_READ_WRITE_TOKEN is not set. Cannot upload files to Vercel Blob.")
                # Fallback to returning local download paths if token is missing
                for local_path_str in generated_files_local_paths:
                    local_path = Path(local_path_str)
                    uploaded_file_urls.append(f"/download-generated/{local_path.name}")
            else:
                for local_path_str in generated_files_local_paths:
                    local_path = Path(local_path_str)
                    try:
                        # Construct the path within Vercel Blob. It's good practice to organize.
                        # E.g., 'generated_exams/20250818_123456_question_paper.pdf'
                        blob_path = f"generated_exams/{local_path.name}" 
                        
                        with open(local_path, 'rb') as f:
                            blob_data = f.read()
                        
                        logger.info(f"Uploading {local_path.name} to Vercel Blob as {blob_path}...")
                        blob = await put(blob_path, blob_data, token=VERCEL_BLOB_READ_WRITE_TOKEN) # Await for async operation
                        uploaded_file_urls.append(blob.url)
                        logger.info(f"Uploaded {local_path.name} to Vercel Blob: {blob.url}")
                    except Exception as upload_error:
                        logger.error(f"❌ Failed to upload {local_path.name} to Vercel Blob: {upload_error}")
                        uploaded_file_urls.append(f"ERROR_UPLOADING_FILE:{local_path.name}") # Indicate individual file upload failure
            
            response_payload = {
                "message": "Exam generation successful",
                "workflow_result": workflow_result,
                "download_urls": uploaded_file_urls # Frontend will use these actual Vercel Blob URLs
            }
            return jsonify(response_payload), 200
        else:
            error_message = workflow_result["workflow_metadata"].get("error", "Unknown error during workflow execution.")
            logger.error(f"API Error: Exam generation failed - {error_message}")
            return jsonify({"error": error_message, "details": workflow_result}), 500

    except Exception as e:
        logger.exception("API Error: An unexpected error occurred") # Logs traceback
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

# Endpoint to serve generated files (for local development/testing ONLY)
# In production, generated files should be served directly from Vercel Blob URLs,
# so this endpoint would typically be removed or made secure.
@app.route('/download-generated/<filename>', methods=['GET'])
def download_generated_file(filename):
    file_path = EXAMS_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    
    mimetype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    
    return send_file(str(file_path), mimetype=mimetype, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    # Ensure generated_exams directory exists for local saving
    EXAMS_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure logs directory for API exists
    api_logs_dir = EXAMS_DIR.parent / "logs"
    api_logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Flask API server...")
    # For async operations (like 'await put'), you might need an ASGI server
    # like Gunicorn with Gevent or Uvicorn, and an async Flask setup (Flask 2.x+).
    # For simple testing, this might work if 'put' handles its own event loop or is mocked.
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True only for development