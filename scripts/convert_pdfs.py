#!/usr/bin/env python3
"""
PDF to Markdown Conversion Script

This script converts all PDF files in the data/input directory to Markdown format
using the MarkerPDF service. It processes files in organized batches and generates
comprehensive reports.

Usage:
    python scripts/convert_pdfs.py [options]

Options:
    --input-dir: Input directory (default: data/input)
    --output-dir: Output directory (default: data/output/converted_markdown)
    --batch-size: Number of files to process in parallel (default: 5)
    --config: Path to configuration file
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.conversion.pdf_converter import PDFConverter
from src.core.utils.logging_config import setup_logging
from config.pdf_conversion_config import get_config


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown using MarkerPDF",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing PDF files (default: data/input)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        help='Output directory for markdown files (default: data/output/converted_markdown)'
    )
    
    parser.add_argument(
        '--kelvin-only',
        action='store_true',
        help='Convert only Kelvin papers'
    )
    
    parser.add_argument(
        '--lectures-only',
        action='store_true', 
        help='Convert only lecture materials'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of files to process in parallel (default: 5)'
    )
    
    parser.add_argument(
        '--max-file-size',
        type=int,
        default=50,
        help='Maximum file size in MB (default: 50)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be converted without actually converting'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def load_configuration(config_path: Optional[str] = None) -> dict:
    """Load configuration from file or use defaults"""
    if config_path and Path(config_path).exists():
        # Load custom configuration
        try:
            import json
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            
            # Merge with default config
            default_config = get_config()
            default_config.update(custom_config)
            return default_config
            
        except Exception as e:
            logging.error(f"Error loading config file {config_path}: {e}")
            return get_config()
    else:
        return get_config()


def print_conversion_summary(results: dict) -> None:
    """Print a formatted summary of conversion results"""
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Session ID: {results['session_id']}")
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    
    if results['failed'] > 0:
        print(f"\nFailed conversions:")
        for result in results['results']:
            if not result['success']:
                print(f"  ✗ {Path(result['input_file']).name}: {result['error']}")
    
    print("="*60)


def convert_kelvin_papers(converter: PDFConverter, config: dict) -> dict:
    """Convert Kelvin exam papers"""
    kelvin_mapping = config['directories']['kelvin_papers']
    
    print(f"\n📚 Converting Kelvin Papers...")
    print(f"Input: {kelvin_mapping['input']}")
    print(f"Output: {kelvin_mapping['output']}")
    
    results = converter.convert_directory(
        str(kelvin_mapping['input']),
        str(kelvin_mapping['output'])
    )
    
    return results


def convert_lectures(converter: PDFConverter, config: dict) -> dict:
    """Convert lecture materials"""
    lectures_mapping = config['directories']['lectures']
    
    print(f"\n🎓 Converting Lecture Materials...")
    print(f"Input: {lectures_mapping['input']}")
    print(f"Output: {lectures_mapping['output']}")
    
    results = converter.convert_directory(
        str(lectures_mapping['input']),
        str(lectures_mapping['output'])
    )
    
    return results


def perform_dry_run(config: dict) -> None:
    """Perform a dry run to show what would be converted"""
    print("\n🔍 DRY RUN - Files that would be converted:")
    print("="*60)
    
    from src.core.conversion.conversion_utils import get_pdf_files
    
    # Check Kelvin papers
    kelvin_dir = config['directories']['kelvin_papers']['input']
    kelvin_files = get_pdf_files(str(kelvin_dir))
    print(f"\nKelvin Papers ({len(kelvin_files)} files):")
    for file in kelvin_files:
        print(f"  📄 {Path(file).name}")
    
    # Check lectures
    lectures_dir = config['directories']['lectures']['input']
    lecture_files = get_pdf_files(str(lectures_dir))
    print(f"\nLecture Materials ({len(lecture_files)} files):")
    for file in lecture_files:
        print(f"  📄 {Path(file).name}")
    
    print(f"\nTotal: {len(kelvin_files) + len(lecture_files)} PDF files")
    print("="*60)


def main():
    """Main conversion function"""
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting PDF to Markdown conversion")
    
    try:
        # Load configuration
        config = load_configuration(args.config)
        
        # Override config with command line arguments
        if args.batch_size:
            config['conversion']['batch_size'] = args.batch_size
        if args.max_file_size:
            config['conversion']['max_file_size_mb'] = args.max_file_size
        
        # Set input/output directories
        input_dir = args.input_dir or str(config['paths']['input'])
        output_dir = args.output_dir or str(config['paths']['output'])
        
        # Perform dry run if requested
        if args.dry_run:
            perform_dry_run(config)
            return
        
        # Initialize converter
        converter = PDFConverter(input_dir, output_dir, config)
        
        # Test API connection
        print("🔗 Testing MarkerPDF connection...")
        success, message = converter.client.test_connection()
        if not success:
            print(f"❌ Connection failed: {message}")
            print("Please check your API key and internet connection.")
            return
        
        print(f"✅ Connection successful: {message}")
        
        # Perform conversions based on options
        all_results = []
        
        if args.kelvin_only:
            results = convert_kelvin_papers(converter, config)
            all_results.append(("Kelvin Papers", results))
        
        elif args.lectures_only:
            results = convert_lectures(converter, config)
            all_results.append(("Lectures", results))
        
        else:
            # Convert both
            kelvin_results = convert_kelvin_papers(converter, config)
            lectures_results = convert_lectures(converter, config)
            
            all_results.append(("Kelvin Papers", kelvin_results))
            all_results.append(("Lectures", lectures_results))
        
        # Print summaries
        for category, results in all_results:
            print(f"\n📊 {category} Results:")
            print_conversion_summary(results)
        
        # Calculate total statistics
        total_files = sum(r[1]['total_files'] for r in all_results)
        total_successful = sum(r[1]['successful'] for r in all_results)
        total_failed = sum(r[1]['failed'] for r in all_results)
        
        print(f"\n🎯 OVERALL SUMMARY:")
        print(f"Total files processed: {total_files}")
        print(f"Successfully converted: {total_successful}")
        print(f"Failed conversions: {total_failed}")
        
        if total_files > 0:
            success_rate = (total_successful / total_files) * 100
            print(f"Overall success rate: {success_rate:.1f}%")
        
        print(f"\n📁 Output directory: {output_dir}")
        print("✅ Conversion process completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Conversion interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.exception("Unexpected error during conversion")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
