#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def debug_content_loading():
    """Debug the content loading process"""
    print("🔍 Debug: Content Loading Process")
    print("=" * 50)
    
    # Check if embeddings file exists
    embeddings_file = Path("data/output/processed/embeddings.json")
    print(f"📁 Embeddings file exists: {embeddings_file.exists()}")
    
    if embeddings_file.exists():
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            print(f"📊 Total embeddings loaded: {len(embeddings_data)}")
            
            if embeddings_data:
                print("\n📝 Sample embedding structure:")
                sample = embeddings_data[0]
                for key, value in sample.items():
                    if key == 'embedding':
                        print(f"  {key}: [array with {len(value)} dimensions]")
                    elif key == 'chunk':
                        print(f"  {key}: '{value[:100]}...' ({len(value)} chars)")
                    else:
                        print(f"  {key}: {value}")
                
                print(f"\n📄 All available chunks:")
                for i, item in enumerate(embeddings_data[:3]):  # Show first 3
                    chunk_text = item.get('chunk', item.get('chunk_text', 'NO CHUNK FOUND'))
                    print(f"  Chunk {i+1}: '{chunk_text[:150]}...'")
                
                return embeddings_data
            else:
                print("❌ Embeddings file is empty")
                return []
                
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return []
    else:
        print("❌ Embeddings file not found")
        return []

def debug_gemini_generation():
    """Debug Gemini API generation with sample content"""
    print("\n🧠 Debug: Gemini API Generation")
    print("=" * 50)
    
    try:
        import google.generativeai as genai
        from config.settings import GEMINI_API_KEY
        
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Test with sample content
        sample_content = """Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."""
        
        prompt = f"""Based on EXACTLY the following educational content, create an intermediate-level multiple choice question.

CONTENT TO USE:
{sample_content}

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
Question: [Your specific question based on the content]
A) [First option]
B) [Second option]  
C) [Third option]
D) [Fourth option]
Correct Answer: [A, B, C, or D]
Explanation: [Why this answer is correct based on the content]"""

        print("📤 Sending prompt to Gemini...")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=600,
            )
        )
        
        print("📥 Gemini response received:")
        print(f"Response text: {response.text}")
        
        return response.text
        
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return None

def debug_question_parsing(response_text):
    """Debug question parsing logic"""
    print("\n🔧 Debug: Question Parsing")
    print("=" * 50)
    
    if not response_text:
        print("❌ No response text to parse")
        return {}
    
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
    print(f"📄 Total lines to parse: {len(lines)}")
    
    question_text = ""
    options = []
    correct_answer = ""
    explanation = ""
    
    for i, line in enumerate(lines):
        print(f"Line {i+1}: '{line}'")
        
        if line.startswith("Question:"):
            question_text = line.replace("Question:", "").strip()
            print(f"  → Found question: '{question_text}'")
        elif line.startswith(("A)", "B)", "C)", "D)")):
            options.append(line)
            print(f"  → Found option: '{line}'")
        elif line.startswith("Correct Answer:"):
            correct_answer = line.replace("Correct Answer:", "").strip()
            print(f"  → Found correct answer: '{correct_answer}'")
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "").strip()
            print(f"  → Found explanation: '{explanation}'")
    
    parsed_question = {
        "type": "multiple_choice",
        "question": question_text,
        "options": options,
        "correct_answer": correct_answer,
        "explanation": explanation
    }
    
    print(f"\n✅ Parsed question structure:")
    for key, value in parsed_question.items():
        print(f"  {key}: {value}")
    
    return parsed_question

if __name__ == "__main__":
    # Run debugging steps
    print("🚀 Starting Exam Generator Debug Process")
    print("=" * 70)
    
    # Step 1: Debug content loading
    content_data = debug_content_loading()
    
    # Step 2: Debug Gemini generation
    response = debug_gemini_generation()
    
    # Step 3: Debug parsing
    if response:
        parsed = debug_question_parsing(response)
        
        if parsed.get('question'):
            print("\n🎉 SUCCESS: Question generation pipeline is working!")
            print("The issue might be in the main exam generator logic.")
        else:
            print("\n❌ FAILURE: Question parsing is not working correctly.")
    
    print("\n" + "=" * 70)
    print("Debug completed. Check the output above to identify the issue.")
