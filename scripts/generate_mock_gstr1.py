"""
Generate mock GSTR-1 summary PDFs matching the standard public layout,
for testing src/gstr1_pdf/parser.py and anomaly_rules.py without a real
filer PDF. Not part of the shipped pipeline — a dev/test utility.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def build_gstr1_pdf(path, gstin, legal_name, return_period, filing_date,
                     taxable_value, igst, cgst, sgst, cess,
                     hsn_rows, doc_total, doc_cancelled, arn):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=A4)
    elements = []

    elements.append(Paragraph("Form GSTR-1", styles["Title"]))
    elements.append(Paragraph("[See rule 59(1)]", styles["Normal"]))
    elements.append(Paragraph("Details of outward supplies of goods or services", styles["Normal"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"1. GSTIN: {gstin}", styles["Normal"]))
    elements.append(Paragraph(f"2. (a) Legal name of the registered person: {legal_name}", styles["Normal"]))
    elements.append(Paragraph(f"Tax Period: {return_period}", styles["Normal"]))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph(
        "3.1 Details of Outward supplies and inward supplies liable to reverse charge",
        styles["Heading3"]))
    table_31_data = [
        ["Nature of Supplies", "Total Taxable value", "Integrated Tax", "Central Tax", "State/UT Tax", "Cess"],
        ["(a) Outward taxable supplies (other than zero rated, nil rated and exempted)",
         f"{taxable_value:,.2f}", f"{igst:,.2f}", f"{cgst:,.2f}", f"{sgst:,.2f}", f"{cess:,.2f}"],
        ["(b) Outward taxable supplies (zero rated)", "0.00", "0.00", "0.00", "0.00", "0.00"],
        ["(c) Other outward supplies (Nil rated, exempted)", "0.00", "0.00", "0.00", "0.00", "0.00"],
        ["(d) Inward supplies (liable to reverse charge)", "0.00", "0.00", "0.00", "0.00", "0.00"],
        ["(e) Non-GST outward supplies", "0.00", "0.00", "0.00", "0.00", "0.00"],
    ]
    t31 = Table(table_31_data, repeatRows=1)
    t31.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))
    elements.append(t31)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("12. HSN-wise summary of outward supplies", styles["Heading3"]))
    hsn_header = ["Sr.No", "HSN", "Description", "UQC", "Total Quantity",
                  "Total value", "Total Taxable value", "Integrated Tax", "Central Tax", "State Tax"]
    hsn_table_data = [hsn_header]
    for i, row in enumerate(hsn_rows, 1):
        hsn_table_data.append([
            str(i), row["hsn"], row["desc"], "NOS",
            f"{row['qty']:,.0f}", f"{row['value']:,.2f}", f"{row['taxable']:,.2f}",
            f"{row['igst']:,.2f}", f"{row['cgst']:,.2f}", f"{row['sgst']:,.2f}",
        ])
    t12 = Table(hsn_table_data, repeatRows=1)
    t12.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))
    elements.append(t12)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("13. Documents issued during the tax period", styles["Heading3"]))
    doc_table_data = [
        ["Nature of Document", "Sr. No From", "Sr. No To", "Total Number", "Cancelled"],
        ["Invoices for outward supply", "INV0001", f"INV{doc_total:04d}", str(doc_total), str(doc_cancelled)],
    ]
    t13 = Table(doc_table_data, repeatRows=1)
    t13.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))
    elements.append(t13)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"ARN: {arn}", styles["Normal"]))
    elements.append(Paragraph(f"Date of filing: {filing_date}", styles["Normal"]))

    doc.build(elements)
    print(f"Built {path}")


if __name__ == "__main__":
    # --- Clean example: tax reconciles at 18%, filed on time, HSN matches ---
    build_gstr1_pdf(
        "/tmp/gstr1_clean.pdf",
        gstin="29ABCDE1234F1Z5", legal_name="Sunrise Traders Pvt Ltd",
        return_period="June-2024", filing_date="09/07/2024",
        taxable_value=500000, igst=0, cgst=45000, sgst=45000, cess=0,
        hsn_rows=[
            {"hsn": "8471", "desc": "Computers", "qty": 100, "value": 350000,
             "taxable": 300000, "igst": 0, "cgst": 27000, "sgst": 27000},
            {"hsn": "8517", "desc": "Telephone sets", "qty": 50, "value": 236000,
             "taxable": 200000, "igst": 0, "cgst": 18000, "sgst": 18000},
        ],
        doc_total=120, doc_cancelled=3, arn="AB240624001234X",
    )

    # --- Anomalous example: tax doesn't reconcile, HSN mismatch, late filing,
    # high cancellation rate ---
    build_gstr1_pdf(
        "/tmp/gstr1_anomalous.pdf",
        gstin="27XYZAB5678G1Z9", legal_name="Quicksilver Trading Co",
        return_period="June-2024", filing_date="15/08/2024",   # ~35 days late
        taxable_value=1000000, igst=0, cgst=8000, sgst=8000, cess=0,  # implies ~1.6% rate, no valid slab
        hsn_rows=[
            {"hsn": "6109", "desc": "T-shirts", "qty": 500, "value": 650000,
             "taxable": 400000, "igst": 0, "cgst": 3200, "sgst": 3200},  # HSN sum << 1,000,000 taxable
        ],
        doc_total=200, doc_cancelled=55, arn="CD240815005678Y",  # 27.5% cancelled
    )
