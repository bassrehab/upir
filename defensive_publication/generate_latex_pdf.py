#!/usr/bin/env python3
"""
Generate professional PDF using LaTeX for UPIR Defensive Publication
"""

import subprocess
import re
from pathlib import Path

def markdown_to_latex(md_content):
    """Convert markdown to LaTeX with proper formatting."""
    
    # Escape LaTeX special characters
    latex = md_content
    latex = latex.replace('\\', '\\textbackslash{}')
    latex = latex.replace('$', '\\$')
    latex = latex.replace('%', '\\%')
    latex = latex.replace('&', '\\&')
    latex = latex.replace('#', '\\#')
    latex = latex.replace('_', '\\_')
    latex = latex.replace('{', '\\{')
    latex = latex.replace('}', '\\}')
    latex = latex.replace('^', '\\textasciicircum{}')
    latex = latex.replace('~', '\\textasciitilde{}')
    
    # Convert markdown headers
    latex = re.sub(r'^# (.+)$', r'\\section{\1}', latex, flags=re.MULTILINE)
    latex = re.sub(r'^## (.+)$', r'\\subsection{\1}', latex, flags=re.MULTILINE)
    latex = re.sub(r'^### (.+)$', r'\\subsubsection{\1}', latex, flags=re.MULTILINE)
    latex = re.sub(r'^#### (.+)$', r'\\paragraph{\1}', latex, flags=re.MULTILINE)
    
    # Convert emphasis
    latex = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', latex)
    latex = re.sub(r'\*(.+?)\*', r'\\textit{\1}', latex)
    latex = re.sub(r'`(.+?)`', r'\\texttt{\1}', latex)
    
    # Convert code blocks
    latex = re.sub(r'```python(.*?)```', r'\\begin{lstlisting}[language=Python]\1\\end{lstlisting}', latex, flags=re.DOTALL)
    latex = re.sub(r'```bash(.*?)```', r'\\begin{lstlisting}[language=bash]\1\\end{lstlisting}', latex, flags=re.DOTALL)
    latex = re.sub(r'```(.*?)```', r'\\begin{lstlisting}\1\\end{lstlisting}', latex, flags=re.DOTALL)
    
    # Convert lists
    latex = re.sub(r'^- (.+)$', r'\\item \1', latex, flags=re.MULTILINE)
    latex = re.sub(r'^(\d+)\. (.+)$', r'\\item \2', latex, flags=re.MULTILINE)
    
    # Handle SVG (replace with placeholder)
    latex = re.sub(r'<svg.*?</svg>', r'\\textit{[Figure: See online version for interactive visualization]}', latex, flags=re.DOTALL)
    
    # Convert tables (simplified)
    lines = latex.split('\n')
    in_table = False
    new_lines = []
    
    for line in lines:
        if '|' in line and not in_table:
            # Start of table
            in_table = True
            cols = line.count('|') - 1
            new_lines.append('\\begin{table}[h]')
            new_lines.append('\\centering')
            new_lines.append('\\begin{tabular}{' + 'l' * cols + '}')
            new_lines.append('\\hline')
            # Process header
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            new_lines.append(' & '.join(cells) + ' \\\\\\\\')
            new_lines.append('\\hline')
        elif '|' in line and '---' in line:
            # Skip separator line
            continue
        elif '|' in line and in_table:
            # Table row
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            new_lines.append(' & '.join(cells) + ' \\\\\\\\')
        elif in_table and '|' not in line:
            # End of table
            in_table = False
            new_lines.append('\\hline')
            new_lines.append('\\end{tabular}')
            new_lines.append('\\end{table}')
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    return '\\n'.join(new_lines)

def generate_latex_document():
    """Generate complete LaTeX document."""
    
    # Read markdown files
    cover_sheet = Path('DISCLOSURE_COVER_SHEET.md').read_text()
    paper = Path('paper_v3_full.md').read_text()
    
    # Convert to LaTeX
    cover_latex = markdown_to_latex(cover_sheet)
    paper_latex = markdown_to_latex(paper)
    
    # Create LaTeX document
    latex_doc = r"""
\documentclass[11pt,a4paper]{article}

% Packages
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{color}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{titlesec}

% Configure hyperref
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    pdftitle={UPIR Defensive Publication},
    pdfauthor={Subhadip Mitra},
}

% Configure listings for code
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.92}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}

\lstset{style=mystyle}

% Headers and footers
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{UPIR Defensive Publication}
\fancyhead[R]{August 2025}
\fancyfoot[C]{Page \thepage\ of \pageref{LastPage}}
\fancyfoot[L]{Google LLC - Confidential}
\fancyfoot[R]{Document ID: UPIR-2025-001}

% Title formatting
\titleformat{\section}{\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{1em}{}

% Document info
\title{\textbf{Universal Plan Intermediate Representation:\\A Practical Framework for Verified Code Generation\\and Compositional System Design}\\[1em]
\large Defensive Publication Disclosure}
\author{Subhadip Mitra\\Google Cloud Professional Services\\subhadip.mitra@google.com}
\date{August 11, 2025}

\begin{document}

% Title page
\maketitle
\thispagestyle{empty}
\newpage

% Table of contents
\tableofcontents
\newpage

% Cover sheet
\section*{DISCLOSURE COVER SHEET}
\addcontentsline{toc}{section}{Disclosure Cover Sheet}
""" + cover_latex + r"""

\newpage

% Main paper
\section*{APPENDIX A: TECHNICAL PAPER}
\addcontentsline{toc}{section}{Appendix A: Technical Paper}
""" + paper_latex + r"""

\end{document}
"""
    
    return latex_doc

def compile_latex(latex_content, output_name="UPIR_Defensive_Publication_2025"):
    """Compile LaTeX to PDF."""
    
    # Write LaTeX file
    tex_file = Path(f"{output_name}.tex")
    tex_file.write_text(latex_content)
    
    print(f"LaTeX file created: {tex_file}")
    print("Compiling to PDF...")
    
    # Compile with pdflatex (run twice for TOC)
    for i in range(2):
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', str(tex_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0 and i == 1:
            print("Warning: LaTeX compilation had some issues.")
            print("Trying simpler compilation...")
            
            # Try with simpler XeLaTeX
            result = subprocess.run(
                ['xelatex', '-interaction=nonstopmode', str(tex_file)],
                capture_output=True,
                text=True
            )
    
    # Check if PDF was created
    pdf_file = Path(f"{output_name}.pdf")
    if pdf_file.exists():
        print(f"✓ PDF generated successfully: {pdf_file}")
        
        # Clean up auxiliary files
        for ext in ['.aux', '.log', '.out', '.toc']:
            aux_file = Path(f"{output_name}{ext}")
            if aux_file.exists():
                aux_file.unlink()
        
        return True
    else:
        print("✗ PDF generation failed")
        print("Check the .log file for errors")
        return False

def main():
    """Main function."""
    print("="*60)
    print("UPIR Defensive Publication - LaTeX PDF Generator")
    print("="*60)
    
    # Check for required files
    if not Path('DISCLOSURE_COVER_SHEET.md').exists():
        print("Error: DISCLOSURE_COVER_SHEET.md not found")
        print("Run this from the defensive_publication/ directory")
        return
    
    if not Path('paper_v3_full.md').exists():
        print("Error: paper_v3_full.md not found")
        return
    
    # Generate LaTeX
    print("\nGenerating LaTeX document...")
    try:
        latex_doc = generate_latex_document()
        
        # Compile to PDF
        if compile_latex(latex_doc):
            print("\n" + "="*60)
            print("SUCCESS! Defensive publication package ready:")
            print("  - UPIR_Defensive_Publication_2025.pdf")
            print("  - UPIR_Defensive_Publication_2025.tex (source)")
            print("\nThis PDF is ready for submission to:")
            print("  - Google Patent Team")
            print("  - Research Disclosure")
            print("  - IP.com")
            print("="*60)
        else:
            print("\nFallback: Use pandoc or online converter")
            print("The .tex file has been created for manual compilation")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTry using pandoc instead:")
        print("  pandoc paper_v3_full.md -o paper_v3.pdf")

if __name__ == "__main__":
    main()