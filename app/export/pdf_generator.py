"""KALESS Engine — PDF Generator.

Produces premium, high-fidelity APA 7 formatted academic reports using ReportLab.
"""
import io
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

def _generate_matplotlib_chart(chart_type: str, data: list, config: dict) -> io.BytesIO:
    """Generates a professional PNG image of the chart using matplotlib with SPSS aesthetics."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # SPSS / Academic styling
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    ax.grid(True, linestyle=':', color='#e0e0e0', axis='y')
    ax.set_axisbelow(True)
    
    # Clean up spines (APA style: no top/right borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    x_label = config.get("x_label", "")
    y_label = config.get("y_label", "")
    
    spss_blue = "#1c4e80" # Deeper, more academic blue
    accent_gray = "#7e909a"
    
    if not data:
        ax.text(0.5, 0.5, "No Data Available for Chart", ha='center', va='center', color='gray')
        ax.set_axis_off()
    elif chart_type in ['bar', 'histogram']:
        names = [str(d.get('name', '')) for d in data]
        values = [d.get('value', 0) for d in data]
        if not values or all(v == 0 for v in values):
             ax.text(0.5, 0.5, "No Values Found", ha='center', va='center', color='gray')
        else:
            color = accent_gray if chart_type == 'histogram' else spss_blue
            bars = ax.bar(names, values, color=color, edgecolor='black', linewidth=0.5, alpha=0.9)
            if chart_type == 'histogram':
                for bar in bars:
                    bar.set_width(1.0)
            plt.xticks(rotation=30, ha='right', fontsize=9)
            
    elif chart_type in ['line', 'area']:
        names = [str(d.get('name', '')) for d in data]
        values = [d.get('value', 0) for d in data]
        if not values:
             ax.text(0.5, 0.5, "No Data Points Found", ha='center', va='center', color='gray')
        else:
            ax.plot(names, values, color=spss_blue, marker='o', linestyle='-', linewidth=1.5, markersize=5, markerfacecolor='white')
            plt.xticks(rotation=30, ha='right', fontsize=9)
            
    elif chart_type == 'scatter':
        xs = [d.get('x') for d in data if 'x' in d]
        ys = [d.get('y') for d in data if 'y' in d]
        if not xs or not ys:
             ax.text(0.5, 0.5, "No X/Y Pairs Found", ha='center', va='center', color='gray')
        else:
            ax.scatter(xs, ys, color=spss_blue, alpha=0.7, edgecolor='black', s=40)
        
    elif chart_type == 'pie':
        names = [str(d.get('name')) for d in data]
        values = [d.get('value', 0) for d in data]
        spss_colors = [spss_blue, "#a5d8ff", "#20639b", "#3caea3", "#f6d55c", "#ed553b"]
        ax.pie(values, labels=names, autopct='%1.1f%%', colors=spss_colors, 
               wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}, textprops={'fontsize': 9})
        ax.grid(False)
        
    if chart_type != 'pie':
        ax.set_xlabel(x_label, fontsize=10, fontstyle='italic')
        ax.set_ylabel(y_label, fontsize=10, fontstyle='italic')
        
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf(result: dict) -> bytes:
    """Generate a premium A4 PDF report with strict APA 7th Edition compliance."""
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72, # 1 inch
        leftMargin=72,  # 1 inch
        topMargin=72,   # 1 inch
        bottomMargin=72, # 1 inch
    )
    
    styles = getSampleStyleSheet()
    
    # --- Custom APA Styles ---
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=14,
        spaceAfter=30,
        alignment=1, # Center
        textTransform='uppercase'
    )
    
    label_style = ParagraphStyle(
        "TableLabel",
        fontName="Times-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=12,
        spaceAfter=2
    )
    
    table_title_style = ParagraphStyle(
        "TableTitle",
        fontName="Times-Italic",
        fontSize=12,
        leading=14,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=12,
        leading=24, # Double spaced
        spaceAfter=12,
        firstLineIndent=36 # Half inch
    )
    
    h1_style = ParagraphStyle(
        "Heading1",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=12,
        alignment=1,
        spaceBefore=24,
        spaceAfter=12
    )
    
    footer_style = ParagraphStyle(
        "Footer",
        fontName="Times-Roman",
        fontSize=8,
        textColor=colors.gray,
        alignment=2 # Right
    )
    
    story = []
    
    # 1. Title Section
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(result.get('title', 'Statistical Analysis Report'), title_style))
    
    timestamp = result.get("metadata", {}).get("timestamp", datetime.now().isoformat())
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        date_str = dt.strftime("%B %d, %Y")
    except:
        date_str = timestamp
        
    story.append(Paragraph(f"<b>Kaless Analysis Engine Output</b>", styles["Normal"]))
    story.append(Paragraph(f"Generated on: {date_str}", styles["Normal"]))
    story.append(Spacer(1, 0.5*inch))
    
    # 2. Iterate Output Blocks
    blocks = result.get("output_blocks", [])
    table_counter = 1
    
    for block in blocks:
        b_type = block.get("block_type")
        title = block.get("title", "Statistical Result")
        content = block.get("content", {})
        
        if b_type == "table":
            # APA 7 Table Label & Title
            story.append(Paragraph(f"Table {table_counter}", label_style))
            story.append(Paragraph(title, table_title_style))
            
            columns = content.get("columns", [])
            rows = content.get("rows", [])
            
            if not columns or not rows:
                story.append(Paragraph("<i>No data available for this table.</i>", body_style))
                table_counter += 1
                continue
                
            matrix = [columns]
            for row in rows:
                matrix.append([str(row.get(col, "") if row.get(col) is not None else ".") for col in columns])
                
            t = Table(matrix, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),
                ('FONTNAME', (0,1), (-1,-1), 'Times-Roman'),
                ('FONTSIZE', (0,0), (-1,-1), 10),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('ALIGN', (0,0), (0,-1), 'LEFT'), 
                # APA Borders: Top, Header-Bottom, Table-Bottom
                ('LINEABOVE', (0,0), (-1,0), 0.5, colors.black),
                ('LINEBELOW', (0,0), (-1,0), 0.5, colors.black),
                ('LINEBELOW', (0,-1), (-1,-1), 0.5, colors.black),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(t)
            
            # Footnotes
            footnotes = content.get("footnotes", [])
            if footnotes:
                note_text = f"<i>Note.</i> " + "; ".join(footnotes)
                story.append(Paragraph(note_text, ParagraphStyle('Note', fontName="Times-Roman", fontSize=9, leading=11)))
            
            story.append(Spacer(1, 0.3*inch))
            table_counter += 1
                
        elif b_type == "chart":
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(Spacer(1, 6))
            
            chart_type = content.get("chart_type", "bar")
            data = content.get("data", [])
            config = content.get("config", {})
            
            try:
                img_buf = _generate_matplotlib_chart(chart_type, data, config)
                img = Image(img_buf, width=6*inch, height=3.8*inch)
                story.append(img)
            except Exception as e:
                story.append(Paragraph(f"[Chart Error: {str(e)}]", styles["Normal"]))
                
            story.append(Spacer(1, 24))
            
        elif b_type == "text":
            story.append(Paragraph(content.get("text", ""), styles["Normal"]))
            story.append(Spacer(1, 12))
            
    # 3. Interpretation
    interp = result.get("interpretation")
    if interp and interp.get("academic_sentence"):
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Reporting (APA Style)", h1_style))
        story.append(Paragraph(interp.get("academic_sentence", ""), body_style))
        
    # 4. Professional Footer
    story.append(Spacer(1, 1*inch))
    footer_text = f"Generated by Kaless Advanced Statistics Engine • https://kaless-web.vercel.app"
    story.append(Paragraph(footer_text, footer_style))

    doc.build(story)
    return buffer.getvalue()
