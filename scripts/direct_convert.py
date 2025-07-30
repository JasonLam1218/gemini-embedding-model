# Create: scripts/direct_convert.py
from pathlib import Path
import pymupdf4llm
import sys

def convert_all_pdfs():
    """Convert all PDFs directly using local pymupdf4llm"""
    print("🚀 Starting PDF to Markdown Conversion")
    print("=" * 60)
    
    # Define directories
    base_input = Path("data/input")
    base_output = Path("data/output/converted_markdown")
    
    # Conversion categories
    categories = {
        "Kelvin Papers": {
            "input": base_input / "kelvin_papers",
            "output": base_output / "kelvin_papers"
        },
        "Lectures": {
            "input": base_input / "lectures", 
            "output": base_output / "lectures"
        }
    }
    
    total_converted = 0
    total_files = 0
    
    # Process each category
    for category_name, paths in categories.items():
        print(f"\n📚 Converting {category_name}...")
        print(f"Input: {paths['input']}")
        print(f"Output: {paths['output']}")
        
        # Create output directory
        paths['output'].mkdir(parents=True, exist_ok=True)
        
        # Get PDF files
        pdf_files = list(paths['input'].glob("*.pdf"))
        total_files += len(pdf_files)
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {paths['input']}")
            continue
        
        print(f"Found {len(pdf_files)} PDF files")
        
        category_success = 0
        for pdf_file in pdf_files:
            try:
                print(f"  📄 Converting: {pdf_file.name}...", end=" ")
                
                # Convert to markdown
                md_content = pymupdf4llm.to_markdown(str(pdf_file))
                
                # Create clean output filename
                clean_name = pdf_file.stem.replace(' ', '_').replace('-', '_')
                output_file = paths['output'] / f"{clean_name}.md"
                
                # Save markdown content
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                
                print("✅")
                category_success += 1
                total_converted += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"  📊 {category_name}: {category_success}/{len(pdf_files)} converted")
    
    # Final summary
    print(f"\n🎯 FINAL RESULTS")
    print("=" * 60)
    print(f"✅ Total converted: {total_converted}/{total_files} files")
    print(f"📁 Output location: {base_output}")
    
    if total_converted > 0:
        success_rate = (total_converted / total_files) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
        print(f"\n📝 Converted files are ready for embedding generation!")
    
    # Create a simple index
    create_simple_index(base_output)

def create_simple_index(base_output):
    """Create a simple README index"""
    try:
        readme_path = base_output / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# Converted PDF Documents\n\n")
            
            # List Kelvin papers
            kelvin_dir = base_output / "kelvin_papers"
            if kelvin_dir.exists():
                f.write("## Kelvin Papers\n\n")
                for md_file in sorted(kelvin_dir.glob("*.md")):
                    title = md_file.stem.replace('_', ' ').title()
                    f.write(f"- [{title}](kelvin_papers/{md_file.name})\n")
                f.write("\n")
            
            # List lectures
            lectures_dir = base_output / "lectures"
            if lectures_dir.exists():
                f.write("## Lectures\n\n")
                for md_file in sorted(lectures_dir.glob("*.md")):
                    title = md_file.stem.replace('_', ' ').title()
                    f.write(f"- [{title}](lectures/{md_file.name})\n")
        
        print(f"📋 Index created: {readme_path}")
        
    except Exception as e:
        print(f"⚠️ Could not create index: {e}")

if __name__ == "__main__":
    convert_all_pdfs()
