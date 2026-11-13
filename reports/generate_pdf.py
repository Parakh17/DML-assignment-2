"""
PDF Generator for Final Report.
Renders markdown to a formatted publication-quality PDF using fpdf2.
"""

import re
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import markdown


class PDFReport(FPDF):
    def header(self):
        if self.page_no() == 1:
            return  # Suppress running header on cover/title page
        self.set_font("Arial", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(100, 6, "DM4ML Assignment 2: Technical Report", align="L")
        self.set_font("Arial", "", 8)
        self.cell(80, 6, "GitHub: Parakh17/DML-assignment-2", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.3)
        self.line(15, 14, 195, 14)
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(226, 232, 240)
        self.set_line_width(0.3)
        self.line(15, 283, 195, 283)
        self.set_font("Arial", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_pdf():
    reports_dir = Path(__file__).resolve().parent
    md_file = reports_dir / "final_report.md"
    pdf_file = reports_dir / "final_report.pdf"
    screenshot_file = reports_dir / "orchestration_run_screenshot.png"

    pdf = PDFReport(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=16)

    # Register Unicode TTF fonts
    font_dir = "/System/Library/Fonts/Supplemental"
    pdf.add_font("Arial", "", f"{font_dir}/Arial.ttf")
    pdf.add_font("Arial", "B", f"{font_dir}/Arial Bold.ttf")
    pdf.add_font("Arial", "I", f"{font_dir}/Arial Italic.ttf")
    pdf.add_font("Courier", "", f"{font_dir}/Courier New.ttf")
    pdf.add_font("Courier", "B", f"{font_dir}/Courier New Bold.ttf")
    pdf.alias_nb_pages()
    pdf.add_page()

    # Read markdown content
    md_text = md_file.read_text(encoding="utf-8")

    # Clean characters incompatible with standard Arial/Courier
    md_text = re.sub(r'[\u2500-\u257f]', '+', md_text)
    box_char_map = {
        "│": "|", "▼": "v", "├": "+", "─": "-", "┌": "+", "┐": "+", "└": "+", "┘": "+", "►": ">",
        "┬": "+", "┴": "+", "┼": "+", "≥": ">=", "≤": "<=", "→": "->", "•": "*", "—": "--", "–": "-"
    }
    for k, v in box_char_map.items():
        md_text = md_text.replace(k, v)

    md_text = md_text.replace("mathbb{R}", "R").replace("\\cdot", "*").replace("\\hat{r}", "r_hat")
    md_text = md_text.replace("$$", "\n\n").replace("$", "")

    # Convert to HTML
    html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    # Enhance HTML styling for FPDF write_html
    html = html.replace("<table>", '<table border="1" width="100%">')
    html = html.replace("<th>", '<th bgcolor="#F1F5F9">')

    # Replace screenshot with local absolute path and format
    if screenshot_file.exists():
        img_tag = f'<center><img src="{screenshot_file}" width="500"></center>'
        html = re.sub(r'<img[^>]*orchestration_run_screenshot\.png[^>]*>', img_tag, html)

    pdf.set_font("Arial", "", 9.5)
    pdf.set_text_color(30, 41, 59)
    pdf.write_html(html)

    pdf.output(str(pdf_file))
    print(f"Successfully generated PDF: {pdf_file}")
    print(f"File size: {pdf_file.stat().st_size:,} bytes across {pdf.page_no()} pages.")


if __name__ == "__main__":
    generate_pdf()
