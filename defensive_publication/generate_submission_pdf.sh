#!/bin/bash

# UPIR Defensive Publication PDF Generator
# This script creates a complete PDF submission package

echo "=================================================="
echo "UPIR Defensive Publication PDF Generator"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "DISCLOSURE_COVER_SHEET.md" ]; then
    echo "Error: Run this script from the defensive_publication/ directory"
    exit 1
fi

# Method 1: Try with pandoc (best quality)
if command -v pandoc &> /dev/null; then
    echo "Using pandoc to generate PDF..."
    
    # Combine cover sheet and paper
    cat DISCLOSURE_COVER_SHEET.md > combined_disclosure.md
    echo -e "\n\n---\n\n# APPENDIX A: TECHNICAL PAPER\n\n" >> combined_disclosure.md
    cat paper_v3_full.md >> combined_disclosure.md
    
    # Generate PDF with nice formatting
    pandoc combined_disclosure.md \
        -o UPIR_Defensive_Publication_2025.pdf \
        --pdf-engine=xelatex \
        --toc \
        --toc-depth=2 \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        -V documentclass=article \
        2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✓ PDF generated successfully: UPIR_Defensive_Publication_2025.pdf"
        rm combined_disclosure.md
        echo ""
        echo "Package ready for submission!"
        echo "Files created:"
        echo "  - UPIR_Defensive_Publication_2025.pdf (Complete disclosure)"
        echo "  - DISCLOSURE_COVER_SHEET.md (Cover sheet)"
        echo "  - paper_v3_full.md (Technical paper)"
        exit 0
    fi
fi

# Method 2: Create HTML for manual conversion
echo "Pandoc not found. Creating HTML version for manual PDF conversion..."

# Create a combined HTML file
cat > combined_disclosure.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>UPIR Defensive Publication</title>
    <style>
        body {
            font-family: 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            max-width: 8.5in;
            margin: 0 auto;
            padding: 1in;
        }
        h1 { 
            font-size: 16pt; 
            margin-top: 24pt;
            page-break-before: auto;
        }
        h2 { font-size: 14pt; margin-top: 18pt; }
        h3 { font-size: 12pt; margin-top: 12pt; }
        pre { 
            background: #f5f5f5; 
            padding: 10px;
            overflow-x: auto;
            font-size: 9pt;
            font-family: 'Courier New', monospace;
        }
        code { 
            background: #f0f0f0;
            padding: 1px 3px;
            font-family: 'Courier New', monospace;
        }
        table { 
            border-collapse: collapse; 
            width: 100%;
            margin: 15px 0;
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 8px;
            text-align: left;
        }
        th { 
            background: #f0f0f0;
            font-weight: bold;
        }
        .page-break {
            page-break-before: always;
        }
        @media print {
            body { margin: 0; }
        }
    </style>
</head>
<body>
EOF

# Convert markdown to HTML (basic conversion)
python3 << 'PYTHON'
import re

def md_to_html(md_file):
    with open(md_file, 'r') as f:
        content = f.read()
    
    # Remove SVG for cleaner conversion
    content = re.sub(r'<svg.*?</svg>', '[FIGURE: See PDF version for visualization]', content, flags=re.DOTALL)
    
    # Basic markdown conversions
    content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
    content = re.sub(r'`(.+?)`', r'<code>\1</code>', content)
    
    # Code blocks
    content = re.sub(r'```python(.*?)```', r'<pre><code>\1</code></pre>', content, flags=re.DOTALL)
    content = re.sub(r'```bash(.*?)```', r'<pre><code>\1</code></pre>', content, flags=re.DOTALL)
    content = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', content, flags=re.DOTALL)
    
    # Line breaks
    content = content.replace('\n\n', '</p><p>')
    content = '<p>' + content + '</p>'
    
    return content

# Process cover sheet
cover_html = md_to_html('DISCLOSURE_COVER_SHEET.md')
with open('combined_disclosure.html', 'a') as f:
    f.write(cover_html)
    f.write('<div class="page-break"></div>')
    f.write('<h1>APPENDIX A: TECHNICAL PAPER</h1>')

# Process main paper
paper_html = md_to_html('paper_v3_full.md')
with open('combined_disclosure.html', 'a') as f:
    f.write(paper_html)

print("HTML file created")
PYTHON

# Close HTML
echo '</body></html>' >> combined_disclosure.html

echo "✓ HTML file created: combined_disclosure.html"
echo ""
echo "=================================================="
echo "TO GENERATE PDF:"
echo "=================================================="
echo ""
echo "Option 1: Open combined_disclosure.html in Chrome/Safari"
echo "         Then: File -> Print -> Save as PDF"
echo ""
echo "Option 2: Install pandoc for automatic generation:"
echo "         brew install pandoc"
echo "         Then run this script again"
echo ""
echo "Option 3: Use online converter:"
echo "         1. Go to https://pandoc.org/try"
echo "         2. Upload paper_v3_full.md"
echo "         3. Convert to PDF"
echo ""
echo "=================================================="