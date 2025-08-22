# gemini-embedding-model/api_controller.py

import os
import json
from flask import Flask, request, jsonify, send_file, abort
from pathlib import Path
from loguru import logger
import io
import mimetypes
from dotenv import load_dotenv
import sys
import re
import requests

from vercel_blob import put # No change here

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.workflows.single_prompt_workflow import SinglePromptWorkflow
# BLOB_READ_WRITE_TOKEN is now correctly imported
from config.settings import EXAMS_DIR, BLOB_READ_WRITE_TOKEN, VERCEL_BLOB_BASE_URL
from src.core.storage.vector_store import VectorStore

app = Flask(__name__)

# Configure logging for the API
logger.remove()
logger.add(sys.stderr, level="INFO")
api_logs_dir = EXAMS_DIR.parent / "logs"
api_logs_dir.mkdir(parents=True, exist_ok=True)
logger.add(f"{api_logs_dir}/api_requests.log", rotation="10 MB", level="INFO")

vector_store = VectorStore()

@app.route('/generate-exam', methods=['POST'])
async def generate_exam_endpoint():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        topic = data.get('topic')
        pdf_blob_urls = data.get('pdf_blob_urls')
        requirements_file = data.get('requirements_file')

        if not topic:
            return jsonify({"error": "Topic is required"}), 400
        if not pdf_blob_urls:
            return jsonify({"error": "PDF Blob URLs are required"}), 400
        
        logger.info(f"API Request: Received request to generate exam for topic '{topic}' with {len(pdf_blob_urls)} PDFs from Vercel Blob.")

        workflow = SinglePromptWorkflow()
        
        workflow_result = await workflow.execute_full_workflow(
            topic=topic,
            requirements_file=requirements_file,
            pdf_blob_urls=pdf_blob_urls
        )

        if workflow_result["workflow_metadata"]["success"]:
            generated_files_local_paths = workflow_result.get("output_files", [])
            
            uploaded_file_urls_for_db = []
            
            # The check for BLOB_READ_WRITE_TOKEN is still important.
            # If it's None, it means the env var is not set, and put() would fail anyway.
            if not BLOB_READ_WRITE_TOKEN:
                logger.error("❌ VERCEL_BLOB_TOKEN is not set. Cannot upload files to Vercel Blob.")
                for local_path_str in generated_files_local_paths:
                    local_path = Path(local_path_str)
                    uploaded_file_urls_for_db.append({
                        "type": "local_fallback",
                        "url": f"/download-generated/{local_path.name}"
                    })
                
                response_payload = {
                    "message": "Exam generation successful (local files only, Vercel Blob upload skipped)",
                    "workflow_result": workflow_result,
                    "download_urls": uploaded_file_urls_for_db
                }
                return jsonify(response_payload), 200

            else:
                for local_path_str in generated_files_local_paths:
                    local_path = Path(local_path_str)
                    file_type_match = re.search(r'comprehensive_([a-zA-Z_]+)_', local_path.name)
                    file_type = file_type_match.group(1) if file_type_match else "unknown_type"

                    try:
                        blob_path = f"generated_exams/{local_path.name}" 
                        
                        with open(local_path, 'rb') as f:
                            blob_data = f.read()
                        
                        logger.info(f"Uploading {local_path.name} to Vercel Blob as {blob_path}...")
                        blob = put(blob_path, blob_data)
                        uploaded_file_urls_for_db.append({
                            "type": file_type, 
                            "url": blob['url'] # <--- MODIFIED THIS LINE
                        })
                        logger.info(f"Uploaded {local_path.name} to Vercel Blob: {blob['url']}") # <--- MODIFIED THIS LINE
                    except Exception as upload_error:
                        logger.error(f"❌ Failed to upload {local_path.name} to Vercel Blob: {upload_error}")
                        uploaded_file_urls_for_db.append({
                            "type": file_type,
                            "url": f"ERROR_UPLOADING_FILE:{local_path.name}"
                        })
                
                if workflow_result.get("generated_papers") and "exam_metadata" in workflow_result["generated_papers"]:
                    workflow_result["generated_papers"]["exam_metadata"]["vercel_blob_urls"] = uploaded_file_urls_for_db
                    workflow_result["download_urls_for_frontend"] = [f_info["url"] for f_info in uploaded_file_urls_for_db if f_info["url"].startswith("http")]

                try:
                    exam_id = vector_store.save_generated_exam(workflow_result["generated_papers"])
                    logger.info(f"✅ Generated exam saved to Supabase with ID: {exam_id}")
                    workflow_result["generated_papers"]["exam_metadata"]["supabase_exam_id"] = exam_id
                except Exception as e:
                    logger.error(f"❌ Failed to save generated exam to Supabase: {e}")
                    workflow_result["workflow_metadata"]["warning"] = f"Failed to save exam to Supabase: {e}"

                response_payload = {
                    "message": "Exam generation and Vercel Blob upload successful",
                    "workflow_result": workflow_result,
                    "download_urls": uploaded_file_urls_for_db
                }
                return jsonify(response_payload), 200
        
        else:
            error_message = workflow_result["workflow_metadata"].get("error", "Unknown error during workflow execution.")
            logger.error(f"API Error: Exam generation failed - {error_message}")
            return jsonify({"error": error_message, "details": workflow_result}), 500

    except Exception as e:
        logger.exception("API Error: An unexpected error occurred")
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

@app.route('/api/download/<int:exam_id>', methods=['GET'])
async def download_exam_file_from_blob(exam_id):
    file_type = request.args.get('type')

    if not file_type:
        return jsonify({"error": "File type is required (e.g., ?type=questions)"}), 400

    logger.info(f"📥 Backend Download Request: examId={exam_id}, type={file_type}")

    try:
        exam_record = vector_store.get_exam_by_id(exam_id)
        if not exam_record:
            logger.error(f"Backend: Exam record not found for ID: {exam_id}")
            abort(404, description="Exam not found")

        exam_json_data = exam_record.get('exam_json')
        if not exam_json_data:
            logger.error(f"Backend: 'exam_json' not found for exam ID: {exam_id}")
            abort(404, description="Exam data corrupted or missing.")
        
        expected_type_key = None
        if file_type == 'questions':
            expected_type_key = 'question_paper'
        elif file_type == 'answers':
            expected_type_key = 'model_answers'
        elif file_type == 'marking':
            expected_type_key = 'marking_scheme'
        else:
            logger.error(f"Backend: Invalid file type requested: {file_type}")
            abort(400, description="Invalid file type requested.")

        vercel_blob_urls = exam_json_data.get('exam_metadata', {}).get('vercel_blob_urls', [])
        
        target_file_url = None
        for file_info in vercel_blob_urls:
            if file_info.get('type') == expected_type_key and file_info.get('url', '').startswith('http'):
                target_file_url = file_info['url']
                break

        if not target_file_url:
            logger.error(f"Backend: Vercel Blob URL not found for exam ID {exam_id}, type {file_type}. Data: {vercel_blob_urls}")
            abort(404, description=f"File '{file_type}' not found for this exam, or URL not available.")

        logger.info(f"Backend: Attempting to download from Vercel Blob: {target_file_url}")
        response = requests.get(target_file_url, stream=True)
        response.raise_for_status()

        file_buffer = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            file_buffer.write(chunk)
        file_buffer.seek(0)

        guessed_mimetype = mimetypes.guess_type(target_file_url)[0] or 'application/pdf'
        download_name = f"exam_{file_type}_{exam_id}.pdf"

        logger.info(f"Backend: Successfully streamed {target_file_url} for exam ID {exam_id}.")
        return send_file(file_buffer, mimetype=guessed_mimetype, as_attachment=True, download_name=download_name)

    except requests.exceptions.RequestException as e:
        logger.error(f"Backend: Error downloading from Vercel Blob URL ({target_file_url if 'target_file_url' in locals() else 'N/A'}): {e}")
        abort(503, description=f"Failed to retrieve file from storage: {e}")
    except Exception as e:
        logger.exception(f"Backend: An unexpected error occurred during download for exam ID {exam_id}, type {file_type}")
        abort(500, description="Internal server error during download process.")

@app.route('/download-generated/<filename>', methods=['GET'])
def download_generated_file(filename):
    file_path = EXAMS_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    
    mimetype = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    
    return send_file(str(file_path), mimetype=mimetype, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    EXAMS_DIR.mkdir(parents=True, exist_ok=True)
    api_logs_dir = EXAMS_DIR.parent / "logs"
    api_logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Flask API server...")
    app.run(host='0.0.0.0', port=5001, debug=True)