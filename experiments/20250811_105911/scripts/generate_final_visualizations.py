#!/usr/bin/env python3
"""
Generate final visualizations for paper v3 from experimental data
"""

import json
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"
VIZ_DIR = EXPERIMENT_DIR / "visualizations"

def load_data():
    """Load experimental results."""
    with open(DATA_DIR / "real_benchmark_results.json", 'r') as f:
        benchmark_data = json.load(f)
    
    with open(RESULTS_DIR / "benchmark_summary.json", 'r') as f:
        summary_data = json.load(f)
    
    # Load learning data from paper
    paper_data_path = Path(__file__).parent.parent.parent.parent / "paper" / "data" / "learning_convergence_results.json"
    with open(paper_data_path, 'r') as f:
        learning_data = json.load(f)
    
    return benchmark_data, summary_data, learning_data

def create_svg_charts():
    """Create SVG charts for the paper."""
    benchmark_data, summary_data, learning_data = load_data()
    
    charts = []
    
    # 1. Code Generation Performance Chart
    templates = benchmark_data['measurements']['code_generation']['templates']
    
    svg_code_gen = f'''<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
  <rect width="600" height="400" fill="white" stroke="black"/>
  <text x="300" y="30" font-size="16" font-weight="bold" text-anchor="middle">Code Generation Performance (Measured)</text>
  <text x="300" y="380" font-size="12" text-anchor="middle">Template</text>
  <text x="30" y="200" font-size="12" text-anchor="middle" transform="rotate(-90 30 200)">Time (ms)</text>
  
  <!-- Grid lines -->
  <g stroke="lightgray" stroke-width="0.5">
    <line x1="80" y1="340" x2="560" y2="340"/>
    <line x1="80" y1="280" x2="560" y2="280"/>
    <line x1="80" y1="220" x2="560" y2="220"/>
    <line x1="80" y1="160" x2="560" y2="160"/>
    <line x1="80" y1="100" x2="560" y2="100"/>
  </g>
  
  <!-- Axes -->
  <line x1="80" y1="60" x2="80" y2="340" stroke="black" stroke-width="2"/>
  <line x1="80" y1="340" x2="560" y2="340" stroke="black" stroke-width="2"/>
  
  <!-- Y-axis labels -->
  <text x="70" y="345" font-size="10" text-anchor="end">0</text>
  <text x="70" y="285" font-size="10" text-anchor="end">0.5</text>
  <text x="70" y="225" font-size="10" text-anchor="end">1.0</text>
  <text x="70" y="165" font-size="10" text-anchor="end">1.5</text>
  <text x="70" y="105" font-size="10" text-anchor="end">2.0</text>
  <text x="70" y="65" font-size="10" text-anchor="end">2.5</text>
  
  <!-- Bars -->'''
    
    x_pos = 100
    bar_width = 70
    for name, time_ms in templates.items():
        bar_height = time_ms * 120  # Scale factor
        y_pos = 340 - bar_height
        
        svg_code_gen += f'''
  <rect x="{x_pos}" y="{y_pos}" width="{bar_width}" height="{bar_height}" fill="steelblue" opacity="0.8"/>
  <text x="{x_pos + bar_width/2}" y="{y_pos - 5}" font-size="10" text-anchor="middle">{time_ms:.2f}ms</text>
  <text x="{x_pos + bar_width/2}" y="355" font-size="9" text-anchor="middle">{name[:8]}</text>'''
        
        x_pos += bar_width + 10
    
    svg_code_gen += '\n</svg>'
    
    # Save the SVG
    with open(VIZ_DIR / "code_generation_performance.svg", 'w') as f:
        f.write(svg_code_gen)
    charts.append("code_generation_performance.svg")
    
    # 2. Verification Speedup Chart
    verification_data = benchmark_data['measurements']['verification']
    
    svg_verification = f'''<svg width="700" height="400" xmlns="http://www.w3.org/2000/svg">
  <rect width="700" height="400" fill="white" stroke="black"/>
  <text x="350" y="30" font-size="16" font-weight="bold" text-anchor="middle">Compositional Verification Speedup (Measured)</text>
  <text x="350" y="380" font-size="12" text-anchor="middle">Number of Components</text>
  <text x="30" y="200" font-size="12" text-anchor="middle" transform="rotate(-90 30 200)">Speedup Factor</text>
  
  <!-- Grid -->
  <g stroke="lightgray" stroke-width="0.5">'''
    
    for i in range(6):
        y = 60 + i * 50
        svg_verification += f'\n    <line x1="80" y1="{y}" x2="650" y2="{y}"/>'
    
    svg_verification += '\n  </g>\n  \n  <!-- Axes -->'
    svg_verification += '\n  <line x1="80" y1="60" x2="80" y2="340" stroke="black" stroke-width="2"/>'
    svg_verification += '\n  <line x1="80" y1="340" x2="650" y2="340" stroke="black" stroke-width="2"/>'
    
    # Plot speedup line
    svg_verification += '\n  \n  <!-- Speedup line -->\n  <polyline fill="none" stroke="green" stroke-width="3" points="'
    
    x_scale = 500 / 60  # Scale to fit 64 components
    y_scale = 280 / 300  # Scale for speedup up to 300x
    
    for item in verification_data:
        x = 80 + item['components'] * x_scale
        y = 340 - item['speedup'] * y_scale
        svg_verification += f'{x},{y} '
    
    svg_verification += '"/>'
    
    # Add data points and labels
    for item in verification_data:
        x = 80 + item['components'] * x_scale
        y = 340 - item['speedup'] * y_scale
        svg_verification += f'\n  <circle cx="{x}" cy="{y}" r="4" fill="green"/>'
        svg_verification += f'\n  <text x="{x}" y="{y-10}" font-size="10" text-anchor="middle">{item["speedup"]:.1f}x</text>'
        svg_verification += f'\n  <text x="{x}" y="355" font-size="10" text-anchor="middle">{item["components"]}</text>'
    
    # Y-axis labels
    for i in range(7):
        speedup = i * 50
        y = 340 - speedup * y_scale
        svg_verification += f'\n  <text x="70" y="{y+5}" font-size="10" text-anchor="end">{speedup}x</text>'
    
    svg_verification += '\n</svg>'
    
    with open(VIZ_DIR / "verification_speedup.svg", 'w') as f:
        f.write(svg_verification)
    charts.append("verification_speedup.svg")
    
    # 3. Learning Convergence Chart (dual axis)
    episodes = learning_data['episodes']
    latencies = learning_data['latencies']
    throughputs = learning_data['throughputs']
    
    svg_learning = f'''<svg width="700" height="400" xmlns="http://www.w3.org/2000/svg">
  <rect width="700" height="400" fill="white" stroke="black"/>
  <text x="350" y="30" font-size="16" font-weight="bold" text-anchor="middle">Learning System Convergence (45 Episodes)</text>
  <text x="350" y="380" font-size="12" text-anchor="middle">Episode</text>
  <text x="30" y="200" font-size="12" fill="red" text-anchor="middle" transform="rotate(-90 30 200)">Latency (ms)</text>
  <text x="670" y="200" font-size="12" fill="green" text-anchor="middle" transform="rotate(90 670 200)">Throughput (req/s)</text>
  
  <!-- Axes -->
  <line x1="80" y1="60" x2="80" y2="340" stroke="black" stroke-width="2"/>
  <line x1="80" y1="340" x2="620" y2="340" stroke="black" stroke-width="2"/>
  <line x1="620" y1="60" x2="620" y2="340" stroke="black" stroke-width="2"/>
  
  <!-- Latency line -->
  <polyline fill="none" stroke="red" stroke-width="2" points="'''
    
    x_scale = 540 / 50  # Scale for 50 episodes
    lat_scale = 280 / 150  # Scale for 50-200ms range
    
    for i in range(0, len(episodes), 2):  # Sample every 2nd point
        x = 80 + episodes[i] * x_scale
        y = 340 - (latencies[i] - 50) * lat_scale
        svg_learning += f'{x},{y} '
    
    svg_learning += '''"/>
  
  <!-- Throughput line -->
  <polyline fill="none" stroke="green" stroke-width="2" points="'''
    
    tput_scale = 280 / 4000  # Scale for 2000-6000 range
    
    for i in range(0, len(episodes), 2):
        x = 80 + episodes[i] * x_scale
        y = 340 - (throughputs[i] - 2000) * tput_scale
        svg_learning += f'{x},{y} '
    
    svg_learning += '''"/>
  
  <!-- Convergence marker -->
  <line x1="535" y1="60" x2="535" y2="340" stroke="orange" stroke-width="2" stroke-dasharray="5,5"/>
  <text x="535" y="55" font-size="10" text-anchor="middle" fill="orange">Converged</text>
  
  <!-- Left Y-axis labels (Latency) -->
  <text x="70" y="345" font-size="10" text-anchor="end" fill="red">50</text>
  <text x="70" y="255" font-size="10" text-anchor="end" fill="red">100</text>
  <text x="70" y="165" font-size="10" text-anchor="end" fill="red">150</text>
  <text x="70" y="75" font-size="10" text-anchor="end" fill="red">200</text>
  
  <!-- Right Y-axis labels (Throughput) -->
  <text x="630" y="345" font-size="10" fill="green">2000</text>
  <text x="630" y="255" font-size="10" fill="green">3000</text>
  <text x="630" y="165" font-size="10" fill="green">4000</text>
  <text x="630" y="75" font-size="10" fill="green">6000</text>
  
  <!-- X-axis labels -->
  <text x="80" y="355" font-size="10" text-anchor="middle">0</text>
  <text x="188" y="355" font-size="10" text-anchor="middle">10</text>
  <text x="296" y="355" font-size="10" text-anchor="middle">20</text>
  <text x="404" y="355" font-size="10" text-anchor="middle">30</text>
  <text x="512" y="355" font-size="10" text-anchor="middle">40</text>
  <text x="620" y="355" font-size="10" text-anchor="middle">50</text>
  
  <!-- Legend -->
  <line x1="250" y1="70" x2="280" y2="70" stroke="red" stroke-width="2"/>
  <text x="285" y="74" font-size="10">Latency</text>
  <line x1="350" y1="70" x2="380" y2="70" stroke="green" stroke-width="2"/>
  <text x="385" y="74" font-size="10">Throughput</text>
</svg>'''
    
    with open(VIZ_DIR / "learning_convergence.svg", 'w') as f:
        f.write(svg_learning)
    charts.append("learning_convergence.svg")
    
    # 4. Synthesis Performance Chart
    synthesis_data = benchmark_data['measurements']['synthesis']
    
    svg_synthesis = f'''<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
  <rect width="600" height="400" fill="white" stroke="black"/>
  <text x="300" y="30" font-size="16" font-weight="bold" text-anchor="middle">Program Synthesis Performance (Measured)</text>
  
  <!-- Two charts side by side -->
  <!-- Left: Time -->
  <text x="150" y="60" font-size="12" font-weight="bold" text-anchor="middle">Synthesis Time</text>
  <g transform="translate(0, 20)">'''
    
    y_pos = 100
    for func_type, data in synthesis_data.items():
        time_ms = data['mean_time_ms']
        bar_width = time_ms * 2  # Scale factor
        
        svg_synthesis += f'''
    <rect x="80" y="{y_pos}" width="{bar_width}" height="30" fill="purple" opacity="0.7"/>
    <text x="75" y="{y_pos + 20}" font-size="10" text-anchor="end">{func_type[:8]}</text>
    <text x="{85 + bar_width}" y="{y_pos + 20}" font-size="10">{time_ms:.1f}ms</text>'''
        
        y_pos += 40
    
    svg_synthesis += '''
  </g>
  
  <!-- Right: Success Rate -->
  <text x="450" y="60" font-size="12" font-weight="bold" text-anchor="middle">Success Rate</text>
  <g transform="translate(300, 20)">'''
    
    y_pos = 100
    for func_type, data in synthesis_data.items():
        success = data['success_rate']
        bar_width = success * 1.5  # Scale factor
        
        svg_synthesis += f'''
    <rect x="80" y="{y_pos}" width="{bar_width}" height="30" fill="orange" opacity="0.7"/>
    <text x="75" y="{y_pos + 20}" font-size="10" text-anchor="end">{func_type[:8]}</text>
    <text x="{85 + bar_width}" y="{y_pos + 20}" font-size="10">{success:.0f}%</text>'''
        
        y_pos += 40
    
    svg_synthesis += '''
  </g>
</svg>'''
    
    with open(VIZ_DIR / "synthesis_performance.svg", 'w') as f:
        f.write(svg_synthesis)
    charts.append("synthesis_performance.svg")
    
    return charts

def main():
    """Generate all visualizations."""
    VIZ_DIR.mkdir(exist_ok=True)
    
    print("Generating visualizations from experimental data...")
    charts = create_svg_charts()
    
    print(f"\nGenerated {len(charts)} charts:")
    for chart in charts:
        print(f"  - {VIZ_DIR / chart}")
    
    print("\nAll visualizations ready for paper v3!")

if __name__ == "__main__":
    main()