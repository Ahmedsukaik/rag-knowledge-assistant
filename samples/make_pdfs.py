"""Rebuild the two sample PDFs:  python samples/make_pdfs.py

The PDFs are committed, so you only need this after editing the text below. They are
text-based PDFs (with a text layer), made with fpdf2. All content is fictional.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF

FIXED_DATE = datetime(2026, 10, 1, tzinfo=timezone.utc)

DOCS = Path(__file__).resolve().parent / "docs"

PRODUCT_MANUAL = {
    "title": "PT-200 Pallet Tracker: User Manual",
    "pages": [
        [
            ("p", "Demo document for Northwind Demo Ltd, a fictional company. The PT-200 is not a real product."),
            ("h", "1. About the PT-200"),
            ("p", "The PT-200 is a handheld scanner that records where each pallet is in the warehouse. "
                  "Scan the pallet label, then scan the shelf label. The location is sent to the stock system over Wi-Fi."),
            ("h", "2. Safety"),
            ("p", "Use the PT-200 only between -20 and 45 degrees Celsius. Do not look into the scanner window "
                  "while it is scanning. Do not open the case; there are no parts you can repair inside."),
        ],
        [
            ("h", "3. Charging the battery"),
            ("p", "Charge the PT-200 in its docking station for 3 hours before first use. "
                  "A full charge lasts about 10 hours of scanning."),
            ("p", "The battery light is green when the battery is full, amber while it is charging, "
                  "and red when less than 15% charge is left."),
            ("p", "Replace the battery after 500 charging cycles or when a full charge lasts less than 6 hours."),
        ],
        [
            ("h", "4. Error codes"),
            ("l", "E12: the scanner window is dirty. Clean it with a dry, soft cloth."),
            ("l", "E31: no Wi-Fi connection. Move closer to an access point, or restart the device."),
            ("l", "E47: battery fault. Stop using the device and contact product support."),
            ("p", "To restart the device, hold the power button for 10 seconds."),
        ],
        [
            ("h", "5. Warranty and support"),
            ("p", "The PT-200 has a 2-year warranty from the date of delivery. The warranty does not cover "
                  "damage from drops or liquids."),
            ("p", "Contact product support at support@northwind-demo.example or on extension 5100. "
                  "Have the serial number ready; it is on the label under the battery."),
        ],
    ],
}

QUARTER_END = {
    "title": "Quarter-End Close Checklist (Finance only)",
    "pages": [
        [
            ("p", "Demo document for Northwind Demo Ltd, a fictional company. This is not a real procedure."),
            ("h", "1. Timeline"),
            ("p", "The quarter-end close must be completed within 8 working days after the last day of the quarter."),
            ("h", "2. Accruals"),
            ("p", "Every cost centre owner sends a list of unbilled costs above 1,000 pounds to finance by working day 2."),
        ],
        [
            ("h", "3. Bank reconciliations"),
            ("p", "All bank accounts are reconciled by working day 3. Differences above 50 pounds are explained in writing."),
            ("h", "4. Sign-off"),
            ("p", "The finance director signs off the management accounts by working day 8. "
                  "The accounts are then shared with the board."),
        ],
    ],
}


class ManualPDF(FPDF):
    def footer(self) -> None:  # a page number at the bottom, as real documents have
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


def build(spec: dict, path: Path) -> None:
    pdf = ManualPDF(format="A4")
    pdf.set_title(spec["title"])
    pdf.set_author("Northwind Demo Ltd (fictional)")
    pdf.set_creation_date(FIXED_DATE)  # a fixed date: the same text gives the same file
    pdf.set_margins(20, 20, 20)
    for number, page in enumerate(spec["pages"], start=1):
        pdf.add_page()
        if number == 1:
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 9, spec["title"], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
        for kind, text in page:
            if kind == "h":
                pdf.ln(3)
                pdf.set_font("Helvetica", "B", 13)
                pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)
            elif kind == "l":
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 6, "- " + text, new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    print(f"wrote {path.relative_to(DOCS.parent.parent)}")


if __name__ == "__main__":
    build(PRODUCT_MANUAL, DOCS / "general" / "pt-200-product-manual.pdf")
    build(QUARTER_END, DOCS / "finance" / "quarter-end-checklist.pdf")
