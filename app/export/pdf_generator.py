"""KALESS Engine — PDF Generator.

Produces formal academic reports using ReportLab and Matplotlib for charts.
"""
import io
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

def _generate_matplotlib_chart(chart_type: str, data: list, config: dict) -> io.BytesIO:
    """Generates a PNG image of the chart using matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # SPSS styling
    ax.set_facecolor('#fdfdfd')
    fig.patch.set_facecolor('#ffffff')
    ax.grid(True, linestyle='--', color='#d4d4d4', axis='y')
    ax.set_axisbelow(True)
    
    x_label = config.get("x_label", "")
    y_label = config.get("y_label", "")
    
    spss_blue = "#4a7bc7"
    
    if chart_type in ['bar', 'histogram']:
        names = [str(d.get('name')) for d in data]
        values = [d.get('value', 0) for d in data]
        color = "#cccccc" if chart_type == 'histogram' else spss_blue
        ax.bar(names, values, color=color, edgecolor='black', linewidth=1)
        if chart_type == 'histogram':
            ax.bar(names, values, width=1.0, color=color, edgecolor='black', linewidth=1)
        else:
            ax.bar(names, values, color=color, edgecolor='black', linewidth=1)
            
        plt.xticks(rotation=45, ha='right')
        
    elif chart_type in ['line', 'area']:
        names = [str(d.get('name')) for d in data]
        values = [d.get('value', 0) for d in data]
        ax.plot(names, values, color=spss_blue, marker='o', linestyle='-', linewidth=2, markersize=4)
        plt.xticks(rotation=45, ha='right')
        
    elif chart_type == 'scatter':
        xs = [d.get('x') for d in data if 'x' in d]
        ys = [d.get('y') for d in data if 'y' in d]
        ax.scatter(xs, ys, color=spss_blue, edgecolor='black')
        
    elif chart_type == 'pie':
        names = [str(d.get('name')) for d in data]
        values = [d.get('value', 0) for d in data]
        spss_colors = [spss_blue, "#800000", "#2e8b57", "#f59e0b", "#8b5cf6", "#ec4899"]
        ax.pie(values, labels=names, autopct='%1.1f%%', colors=spss_colors, 
               wedgeprops={'edgecolor': 'black'})
        ax.grid(False)
        
    if chart_type != 'pie':
        ax.set_xlabel(x_label, fontweight='bold', fontsize=10)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf(result: dict) -> bytes:
    """Generate an APA-formatted A4 PDF from a NormalizedResult dictionary."""
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        spaceAfter=20,
        alignment=1 # Center
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        spaceBefore=15,
        spaceAfter=10
    )
    normal_style = styles["Normal"]
    
    interp_style = ParagraphStyle(
        "Interpretation",
        parent=styles["Normal"],
        fontName="Times-Roman", # APA prefers Times New Roman
        fontSize=12,
        leading=24, # Double spaced
        spaceAfter=12,
        firstLineIndent=36 # Half inch indent
    )
    
    story = []
    
    # 1. Cover / Title
    story.append(Paragraph("KALESS Statistics Report", title_style))
    story.append(Paragraph(f"<b>Analysis:</b> {result.get('title', 'Statistical Analysis')}", normal_style))
    story.append(Spacer(1, 12))
    
    var_str = ", ".join([f"{k}: {v}" for k, v in result.get("variables", {}).items()])
    story.append(Paragraph(f"<b>Variables:</b> {var_str}", normal_style))
    story.append(Spacer(1, 24))
    
    # 2. Iterate Output Blocks
    blocks = result.get("output_blocks", [])
    
    for block in blocks:
        b_type = block.get("block_type")
        title = block.get("title", "")
        content = block.get("content", {})
        
        if b_type == "table":
            story.append(Paragraph(title, h2_style))
            
            columns = content.get("columns", [])
            rows = content.get("rows", [])
            
            if not columns or not rows:
                continue
                
            # Build matrix
            matrix = [columns]
            for row in rows:
                matrix.append([str(row.get(col, "")) for col in columns])
                
            # APA 7th Edition Table Style
            # 1. Horizontal lines above and below headers, and below bottom row.
            # 2. No vertical lines.
            t = Table(matrix, repeatRows=1)
            t.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
                ('FONTNAME', (0,1), (-1,-1), 'Times-Roman'),
                ('FONTSIZE', (0,0), (-1,-1), 10),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('ALIGN', (0,0), (0,-1), 'LEFT'), # First column left aligned usually
                ('LINEABOVE', (0,0), (-1,0), 1, colors.black),
                ('LINEBELOW', (0,0), (-1,0), 1, colors.black),
                ('LINEBELOW', (0,-1), (-1,-1), 1, colors.black),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(t)
            story.append(Spacer(1, 15))
            
            # Footnotes
            footnotes = content.get("footnotes", [])
            for fn in footnotes:
                story.append(Paragraph(f"<i>Note.</i> {fn}", ParagraphStyle('Note', fontName="Times-Italic", fontSize=10)))
            if footnotes:
                story.append(Spacer(1, 15))
                
        elif b_type == "chart":
            story.append(Paragraph(title, h2_style))
            story.append(Spacer(1, 6))
            
            chart_type = content.get("chart_type", "bar")
            data = content.get("data", [])
            config = content.get("config", {})
            
            try:
                img_buf = _generate_matplotlib_chart(chart_type, data, config)
                # 6x4 inches -> proportional scaling
                img = Image(img_buf, width=5.5*inch, height=3.66*inch)
                story.append(img)
            except Exception as e:
                story.append(Paragraph(f"[Chart generation failed: {str(e)}]", normal_style))
                
            story.append(Spacer(1, 20))
            
        elif b_type == "text":
            story.append(Paragraph(title, h2_style))
            story.append(Paragraph(content.get("text", ""), normal_style))
            story.append(Spacer(1, 15))
            
    # 3. Interpretation (APA write-up)
    interp = result.get("interpretation")
    if interp and interp.get("academic_sentence"):
        story.append(Paragraph("Reporting (APA Style)", h2_style))
        story.append(Paragraph(interp.get("academic_sentence", ""), interp_style))
        
    # 4. Metadata Footer
    story.append(Spacer(1, 30))
    meta = result.get("metadata", {})
    footer_text = f"Generated by KALESS Engine on {meta.get('timestamp', 'Unknown Date')}"
    story.append(Paragraph(f"<font color='gray' size='8'>{footer_text}</font>", normal_style))

    doc.build(story)
    
    return buffer.getvalue()
