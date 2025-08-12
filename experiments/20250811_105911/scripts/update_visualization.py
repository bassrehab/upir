#!/usr/bin/env python3
"""
Update code generation performance visualization with actual measured values.
"""

def create_code_gen_svg():
    """Create SVG for code generation performance with actual values."""
    
    # Actual measured values
    templates = {
        'Queue Worker': 2.83,
        'Rate Limiter': 1.60,
        'Circuit Breaker': 1.51,
        'Retry Logic': 1.46,
        'Cache': 1.47,
        'Load Balancer': 1.37
    }
    
    # SVG dimensions
    width = 600
    height = 400
    margin = 80
    bar_width = 70
    bar_spacing = 10
    
    # Calculate positions
    chart_height = height - 2 * margin
    max_value = 3.0  # Maximum ms for scaling
    
    svg_lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'  <rect width="{width}" height="{height}" fill="white" stroke="black"/>',
        '  <text x="300" y="30" font-size="16" font-weight="bold" text-anchor="middle">Code Generation Performance (Actual Measurements)</text>',
        '  <text x="300" y="380" font-size="12" text-anchor="middle">Template Type</text>',
        '  <text x="30" y="200" font-size="12" text-anchor="middle" transform="rotate(-90 30 200)">Generation Time (ms)</text>',
    ]
    
    # Add grid lines
    svg_lines.append('  <g stroke="lightgray" stroke-width="0.5">')
    for i in range(4):
        y = margin + i * chart_height / 3
        svg_lines.append(f'    <line x1="{margin}" y1="{y}" x2="{width-margin+20}" y2="{y}"/>')
        # Add y-axis labels
        value = 3.0 - i * 1.0
        svg_lines.append(f'    <text x="{margin-10}" y="{y+5}" font-size="10" text-anchor="end">{value:.1f}ms</text>')
    svg_lines.append('  </g>')
    
    # Add axes
    svg_lines.append(f'  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black" stroke-width="2"/>')
    svg_lines.append(f'  <line x1="{margin}" y1="{height-margin}" x2="{width-margin+20}" y2="{height-margin}" stroke="black" stroke-width="2"/>')
    
    # Add bars
    x_pos = margin + 20
    for template, time_ms in templates.items():
        bar_height = (time_ms / max_value) * chart_height
        bar_y = height - margin - bar_height
        
        # Bar
        svg_lines.append(f'  <rect x="{x_pos}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="steelblue" opacity="0.8"/>')
        
        # Value label
        svg_lines.append(f'  <text x="{x_pos + bar_width/2}" y="{bar_y - 5}" font-size="10" text-anchor="middle">{time_ms:.2f}ms</text>')
        
        # Template name
        svg_lines.append(f'  <text x="{x_pos + bar_width/2}" y="{height-margin+20}" font-size="9" text-anchor="middle" transform="rotate(-45 {x_pos + bar_width/2} {height-margin+20})">{template}</text>')
        
        x_pos += bar_width + bar_spacing
    
    # Add average line
    avg = sum(templates.values()) / len(templates)
    avg_y = height - margin - (avg / max_value) * chart_height
    svg_lines.append(f'  <line x1="{margin}" y1="{avg_y}" x2="{width-margin+20}" y2="{avg_y}" stroke="red" stroke-width="1" stroke-dasharray="5,5"/>')
    svg_lines.append(f'  <text x="{width-margin}" y="{avg_y-5}" font-size="10" text-anchor="end" fill="red">Avg: {avg:.2f}ms</text>')
    
    svg_lines.append('</svg>')
    
    return '\n'.join(svg_lines)


if __name__ == "__main__":
    svg_content = create_code_gen_svg()
    
    # Save to file
    output_path = '../visualizations/code_generation_performance_actual.svg'
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    print(f"Updated visualization saved to {output_path}")
    print("\nActual measured values:")
    print("- Queue Worker: 2.83ms")
    print("- Rate Limiter: 1.60ms")
    print("- Circuit Breaker: 1.51ms")
    print("- Retry Logic: 1.46ms")
    print("- Cache: 1.47ms")
    print("- Load Balancer: 1.37ms")
    print(f"- Average: 1.71ms")