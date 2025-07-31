#!/usr/bin/env python3
"""
Simplified script for one-command paper generation.
This is your main entry point for the streamlined workflow.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.workflows.single_prompt_workflow import SinglePromptWorkflow
from loguru import logger

def main():
    """One-command paper generation"""
    parser = argparse.ArgumentParser(description='Generate comprehensive exam papers')
    parser.add_argument('--topic', required=True, help='Exam topic (e.g., "AI and Data Analytics")')
    parser.add_argument('--requirements', help='Custom requirements file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.add(sys.stdout, level="DEBUG")
    
    logger.info("🚀 Starting One-Command Paper Generation")
    logger.info(f"📚 Topic: {args.topic}")
    
    try:
        # Initialize workflow
        workflow = SinglePromptWorkflow()
        
        # Execute complete workflow
        result = workflow.execute_full_workflow(args.topic, args.requirements)
        
        if result["workflow_metadata"]["success"]:
            print("\n✅ SUCCESS! Generated papers are ready:")
            print(f"📁 Location: data/output/generated_papers/")
            print(f"⏱️  Total time: {result['workflow_metadata']['duration_seconds']:.1f} seconds")
            
            # List generated files
            output_files = result.get('output_files', [])
            for file_path in output_files:
                file_name = Path(file_path).name
                print(f"   📄 {file_name}")
        else:
            print(f"\n❌ FAILED: {result['workflow_metadata'].get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
