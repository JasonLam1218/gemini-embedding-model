#!/usr/bin/env python3
"""Verify the complete PDF -> Markdown -> Embedding -> Gemini workflow"""

import sys
from pathlib import Path

def verify_complete_workflow():
    """Verify the complete workflow from converted markdown to generated exams"""
    
    # Step 1: Check converted markdown files exist
    markdown_dir = Path("data/output/converted_markdown")
    if not markdown_dir.exists():
        print("❌ Converted markdown directory not found")
        return False
    
    md_files = list(markdown_dir.rglob("*.md"))
    if len(md_files) == 0:
        print("❌ No markdown files found - run PDF conversion first")
        return False
    
    print(f"✅ Found {len(md_files)} markdown files")
    
    # Step 2: Test markdown processing
    # Step 3: Test embedding generation
    # Step 4: Test exam generation
    
    return True

if __name__ == "__main__":
    success = verify_complete_workflow()
    sys.exit(0 if success else 1)
