#!/usr/bin/env python3
"""
Generate PDF from paper_v3.md
Requires: pip install markdown2 weasyprint
Alternative: Can use pandoc if available
"""

import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required tools are installed."""
    tools = {
        'pandoc': 'brew install pandoc',
        'wkhtmltopdf': 'brew install --cask wkhtmltopdf',
    }
    
    available = {}
    for tool, install_cmd in tools.items():
        try:
            subprocess.run([tool, '--version'], capture_output=True, check=True)
            available[tool] = True
            print(f"✓ {tool} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            available[tool] = False
            print(f"✗ {tool} not found. Install with: {install_cmd}")
    
    return available

def generate_with_pandoc():
    """Generate PDF using pandoc (best quality)."""
    print("\nGenerating PDF with pandoc...")
    
    # Create a simple style file for better formatting
    latex_header = """\\usepackage{geometry}
\\geometry{a4paper, margin=1in}
\\usepackage{hyperref}
\\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}
\\usepackage{fancyhdr}
\\pagestyle{fancy}
\\fancyhead[L]{UPIR Technical Disclosure}
\\fancyhead[R]{August 2025}
\\usepackage{listings}
\\lstset{basicstyle=\\ttfamily\\small,breaklines=true}
"""
    
    # Write header to temp file
    header_file = Path("paper_header.tex")
    header_file.write_text(latex_header)
    
    cmd = [
        'pandoc',
        'paper_v3.md',
        '-o', 'paper_v3.pdf',
        '--pdf-engine=xelatex',
        '-H', 'paper_header.tex',
        '--toc',
        '--toc-depth=2',
        '--highlight-style=kate',
        '-V', 'documentclass=article',
        '-V', 'fontsize=11pt',
        '-V', 'papersize=a4',
        '-V', 'colorlinks=true'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ PDF generated successfully: paper_v3.pdf")
        header_file.unlink()  # Clean up temp file
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating PDF: {e}")
        return False

def generate_with_markdown_and_wkhtmltopdf():
    """Generate PDF using markdown2 and wkhtmltopdf."""
    print("\nGenerating PDF with markdown2 and wkhtmltopdf...")
    
    try:
        import markdown2
    except ImportError:
        print("Installing markdown2...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'markdown2'])
        import markdown2
    
    # Read markdown
    md_content = Path('paper_v3.md').read_text()
    
    # Convert to HTML
    html = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks'])
    
    # Add CSS for better formatting
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>UPIR Technical Disclosure</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                font-size: 11pt;
                line-height: 1.5;
                max-width: 8.5in;
                margin: 1in auto;
                padding: 0;
            }}
            h1 {{ font-size: 16pt; margin-top: 24pt; }}
            h2 {{ font-size: 14pt; margin-top: 18pt; }}
            h3 {{ font-size: 12pt; margin-top: 12pt; }}
            pre {{ 
                background: #f4f4f4; 
                padding: 10px;
                overflow-x: auto;
                font-size: 9pt;
            }}
            code {{ 
                background: #f0f0f0;
                padding: 2px 4px;
                font-size: 9pt;
            }}
            table {{ 
                border-collapse: collapse; 
                width: 100%;
                margin: 10px 0;
            }}
            th, td {{ 
                border: 1px solid #ddd; 
                padding: 8px;
                text-align: left;
            }}
            th {{ background: #f0f0f0; }}
            svg {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # Write HTML
    html_file = Path('paper_v3.html')
    html_file.write_text(html_with_style)
    
    # Convert to PDF
    cmd = ['wkhtmltopdf', '--enable-local-file-access', 'paper_v3.html', 'paper_v3.pdf']
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ PDF generated successfully: paper_v3.pdf")
        html_file.unlink()  # Clean up
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating PDF: {e}")
        return False

def generate_simple_text_version():
    """Generate a simple text version for review."""
    print("\nGenerating text version for easy review...")
    
    md_content = Path('paper_v3.md').read_text()
    
    # Remove SVG content for cleaner text
    import re
    text_content = re.sub(r'<svg.*?</svg>', '[FIGURE: See PDF for visualization]', md_content, flags=re.DOTALL)
    
    # Save as text
    Path('paper_v3.txt').write_text(text_content)
    print("✓ Text version saved: paper_v3.txt")

def main():
    print("="*60)
    print("UPIR Paper v3 PDF Generator")
    print("="*60)
    
    # Check what tools are available
    available = check_dependencies()
    
    # Try pandoc first (best quality)
    if available['pandoc']:
        if generate_with_pandoc():
            generate_simple_text_version()
            print("\n✓ Success! Files generated:")
            print("  - paper_v3.pdf (high quality PDF)")
            print("  - paper_v3.txt (text version)")
            return
    
    # Fallback to wkhtmltopdf
    if available['wkhtmltopdf']:
        if generate_with_markdown_and_wkhtmltopdf():
            generate_simple_text_version()
            print("\n✓ Success! Files generated:")
            print("  - paper_v3.pdf")
            print("  - paper_v3.txt")
            return
    
    # If no tools available, provide instructions
    print("\n" + "="*60)
    print("No PDF generation tools found. Please install one of:")
    print("  Option 1: brew install pandoc")
    print("  Option 2: brew install --cask wkhtmltopdf")
    print("\nAlternatively, you can:")
    print("  1. Copy paper_v3.md to a Markdown editor")
    print("  2. Export as PDF from the editor")
    print("  3. Or use online converters like pandoc.org/try")
    print("="*60)

if __name__ == "__main__":
    main()