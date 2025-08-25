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
import subprocess
import asyncio    # Not strictly used for subprocess but good to keep if needed elsewhere
from datetime import datetime # NEW: Import datetime for exam_data_for_supabase fallback
import time       # NEW: Import time for small delay

from vercel_blob import put

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import EXAMS_DIR, BLOB_READ_WRITE_TOKEN, MARKDOWN_INPUT_DIR # MARKDOWN_INPUT_DIR might not be strictly needed here, but kept.
from src.core.storage.vector_store import VectorStore 

app = Flask(__name__)

# Configure logging for the API
logger.remove()
logger.add(sys.stderr, level="INFO")
api_logs_dir = EXAMS_DIR.parent / "logs"
api_logs_dir.mkdir(parents=True, exist_ok=True)
logger.add(f"{api_logs_dir}/api_requests.log", rotation="10 MB", level="INFO")

vector_store = VectorStore()

# Helper function to run shell commands and check for errors
def run_command(command_list, description="command"):
    logger.info(f"Executing: {' '.join(command_list)}")
    try:
        # Using sys.executable ensures the command runs with the same Python interpreter
        # that is running the Flask app, which is crucial in virtual environments.
        result = subprocess.run([sys.executable] + command_list, capture_output=True, text=True, check=True)
        logger.info(f"✅ {description} successful.")
        logger.debug(f"STDOUT: {result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}.")
        logger.error(f"STDERR: {e.stderr}")
        logger.error(f"STDOUT: {e.stdout}") 
        raise RuntimeError(f"Command '{' '.join(command_list)}' failed: {e.stderr}")
    except FileNotFoundError:
        logger.error(f"❌ Python interpreter or script '{command_list[0]}' not found. Ensure Python and scripts are in PATH.")
        raise RuntimeError(f"Python interpreter or script not found. Check environment setup.")
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during {description}: {e}")
        raise RuntimeError(f"Unexpected error during {description}: {e}")

@app.route('/generate-exam', methods=['POST'])
async def generate_exam_endpoint():
    start_api_call_time = time.time()
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        topic = data.get('topic')
        # pdf_blob_urls = data.get('pdf_blob_urls') # This will be ignored for subprocess.run('direct_convert.py')

        if not topic:
            return jsonify({"error": "Topic is required"}), 400

        logger.info(f"API Request: Received request to generate exam for topic '{topic}'. Using external commands.")
        
        # Ensure EXAMS_DIR exists (it's created in __init__ but good practice to ensure before use)
        EXAMS_DIR.mkdir(parents=True, exist_ok=True)

        # Store initial files in EXAMS_DIR before generation to help identify new ones
        initial_files_in_exams_dir = set(p.name for p in EXAMS_DIR.iterdir() if p.is_file())
        logger.debug(f"Files in EXAMS_DIR before generation: {initial_files_in_exams_dir}")
        
        # Step 1: Run direct_convert.py
        # WARNING: This command, when run directly as a subprocess, will likely
        # process ALL PDFs from Vercel Blob or local 'data/input'.
        # It will NOT be limited to specific 'pdf_blob_urls' passed to this API endpoint.
        logger.info("STEP 1: Converting PDFs to Markdown...")
        success, _ = run_command(["scripts/direct_convert.py"], "PDF conversion")
        if not success:
            raise RuntimeError("PDF conversion failed.")
        
        # Step 2: Run process-texts
        logger.info("STEP 2: Processing texts...")
        success, _ = run_command(["run_pipeline.py", "process-texts", "--use-supabase"], "Text processing")
        if not success:
            raise RuntimeError("Text processing failed.")

        # Step 3: Run generate-embeddings
        logger.info("STEP 3: Generating embeddings...")
        success, _ = run_command(["run_pipeline.py", "generate-embeddings", "--use-supabase"], "Embedding generation")
        if not success:
            raise RuntimeError("Embedding generation failed.")

        # Step 4: Run generate-comprehensive-papers
        logger.info("STEP 4: Generating comprehensive papers...")
        success, _ = run_command(["run_pipeline.py", "generate-comprehensive-papers", "--topic", topic], "Paper generation")
        if not success:
            raise RuntimeError("Paper generation failed.")

        # --- Post-generation: Infer generated files and handle Vercel Blob upload / Supabase save ---
        # Introduce a small delay to ensure all files are written to disk
        time.sleep(2) # Added 2-second delay
        
        # Get all current files in EXAMS_DIR after generation
        current_files_in_exams_dir = set(p.name for p in EXAMS_DIR.iterdir() if p.is_file())
        logger.debug(f"Files in EXAMS_DIR after generation: {current_files_in_exams_dir}")

        # Identify newly generated files based on what was there before
        # Filter for expected suffixes (PDF and JSON)
        newly_generated_filenames = [
            f for f in (current_files_in_exams_dir - initial_files_in_exams_dir)
            if f.endswith(('.pdf', '.json')) and f.startswith('comprehensive_') # Assuming your naming convention
        ]
        
        generated_files_local_paths = [EXAMS_DIR / fn for fn in newly_generated_filenames]

        # Filter out temporary/incomplete files if any (e.g., from partial writes)
        # Only include files that exist and have a substantial size
        generated_files_local_paths = [p for p in generated_files_local_paths if p.exists() and p.stat().st_size > 100] # Min size check
        
        logger.info(f"Identified {len(generated_files_local_paths)} new files for upload: {[p.name for p in generated_files_local_paths]}")

        if not generated_files_local_paths:
            logger.error("❌ No new files found in generated_exams directory after pipeline execution, or files were too small.")
            workflow_result_metadata = {"success": False, "error": "No valid output files generated by pipeline."}
            return jsonify({"error": "Failed to find generated output files.", "workflow_metadata": workflow_result_metadata}), 500

        # Now proceed with Vercel Blob upload and Supabase save, similar to original logic
        uploaded_file_urls_for_db = []
        
        if not BLOB_READ_WRITE_TOKEN:
            logger.error("❌ VERCEL_BLOB_TOKEN is not set. Cannot upload files to Vercel Blob. Providing local fallback URLs.")
            for local_path in generated_files_local_paths:
                uploaded_file_urls_for_db.append({
                    "type": "local_fallback",
                    "url": f"/download-generated/{local_path.name}"
                })
            
            response_payload = {
                "message": "Exam generation successful (local files only, Vercel Blob upload skipped)",
                "workflow_result": {"workflow_metadata": {"success": True, "topic": topic}, "output_files": [str(p) for p in generated_files_local_paths]}, 
                "download_urls": uploaded_file_urls_for_db
            }
            return jsonify(response_payload), 200

        else:
            # We need a placeholder for exam_data that typically comes from workflow_result["generated_papers"]
            # Attempt to find and load the comprehensive JSON file for Supabase save
            exam_data_for_supabase = {"exam_metadata": {"topic": topic, "generated_at": datetime.now().isoformat()}} # Default minimal
            json_files_found = [p for p in generated_files_local_paths if p.suffix == '.json' and 'complete_exam' in p.name]
            
            if json_files_found:
                try:
                    with open(json_files_found[0], 'r', encoding='utf-8') as f:
                        exam_data_for_supabase = json.load(f)
                    logger.info(f"Loaded comprehensive JSON for Supabase: {json_files_found[0].name}")
                except Exception as e:
                    logger.warning(f"Could not load comprehensive JSON {json_files_found[0].name} for Supabase: {e}. Using minimal metadata for Supabase save.")
            else:
                 logger.warning("No comprehensive JSON file found among generated outputs. Using minimal metadata for Supabase save.")

            for local_path in generated_files_local_paths:
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
                        "url": blob['url']
                    })
                    logger.info(f"Uploaded {local_path.name} to Vercel Blob: {blob['url']}")
                except Exception as upload_error:
                    logger.error(f"❌ Failed to upload {local_path.name} to Vercel Blob: {upload_error}")
                    uploaded_file_urls_for_db.append({
                        "type": file_type,
                        "url": f"ERROR_UPLOADING_FILE:{local_path.name}"
                    })
            
            # Update exam_data_for_supabase with Vercel Blob URLs for persistent record
            if "exam_metadata" not in exam_data_for_supabase:
                exam_data_for_supabase["exam_metadata"] = {}
            exam_data_for_supabase["exam_metadata"]["vercel_blob_urls"] = uploaded_file_urls_for_db

            # Now save to Supabase
            try:
                exam_id = vector_store.save_generated_exam(exam_data_for_supabase)
                logger.info(f"✅ Generated exam saved to Supabase with ID: {exam_id}")
                if "exam_metadata" not in exam_data_for_supabase:
                    exam_data_for_supabase["exam_metadata"] = {} # Ensure it's a dict
                exam_data_for_supabase["exam_metadata"]["supabase_exam_id"] = exam_id
            except Exception as e:
                logger.error(f"❌ Failed to save generated exam to Supabase: {e}")
                # Add warning to the metadata for the response
                if "workflow_metadata" not in exam_data_for_supabase:
                    exam_data_for_supabase["workflow_metadata"] = {}
                exam_data_for_supabase["workflow_metadata"]["warning"] = f"Failed to save exam to Supabase: {e}"

            response_payload = {
                "message": "Exam generation and Vercel Blob upload successful",
                # Mimic the structure of workflow_result for the frontend response
                "workflow_result": {
                    "workflow_metadata": {"success": True, "topic": topic, "duration_seconds": round(time.time() - start_api_call_time, 2)},
                    "generated_papers": exam_data_for_supabase,
                    "output_files": [str(p) for p in generated_files_local_paths],
                    "download_urls_for_frontend": [f_info["url"] for f_info in uploaded_file_urls_for_db if f_info["url"].startswith("http")]
                },
                "download_urls": uploaded_file_urls_for_db # Provided for compatibility if frontend uses this top-level key
            }
            return jsonify(response_payload), 200

    except RuntimeError as e: # Catch errors raised by run_command and other explicit raises
        logger.error(f"API Error: External command execution failed - {e}")
        workflow_result_metadata = {"success": False, "error": str(e)}
        return jsonify({"error": str(e), "details": "Check backend logs for subprocess errors.", "workflow_metadata": workflow_result_metadata}), 500
    except Exception as e:
        logger.exception("API Error: An unexpected internal error occurred")
        workflow_result_metadata = {"success": False, "error": f"An internal server error occurred: {str(e)}"}
        return jsonify({"error": "An internal server error occurred", "details": str(e), "workflow_metadata": workflow_result_metadata}), 500

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