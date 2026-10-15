"""
GSTR-1 PDF parser.

Real taxpayers file GSTR-1 through the GST portal and typically download
it as a PDF (or JSON) summary — not the anonymized Column0..Column21 CSV
schema the trained GNN model uses. There is no honest way to map GSTR-1
fields onto those anonymized columns (we don't know what they represent),
so this module is deliberately independent of src/phase3_pipeline — it
extracts real, named GSTR-1 fields for the rule-based anomaly checks in
gstr1_pdf/anomaly_rules.py instead.

Targets the standard GSTR-1 summary PDF layout (Form GSTR-1, [See rule
59(1)]) as published on the GST portal:
    - Header: GSTIN, legal name, return period
    - Table 3.1: outward supplies summary (taxable value, IGST, CGST,
      SGST/UTGST, Cess) broken down by supply nature
    - Table 12: HSN-wise summary of outward supplies
    - Table 13: documents issued summary (invoice counts, cancellations)
    - Footer: ARN and filing date

Caveat, stated plainly: this was built and tested against a
programmatically-generated mock GSTR-1 PDF matching the documented public
layout (no real filer PDF was available to validate against). Real PDFs
exported by different GST portal versions or third-party filing software
may have layout variations that need adjusting the regex/table-matching
below — see _extract_table_31 / _extract_table_12 / _extract_table_13 if
extraction comes back empty or wrong for a real file.
"""

import re
import pdfplumber
import numpy as np


GSTIN_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]\b")
ARN_PATTERN = re.compile(r"ARN[:\s]+([A-Z]{2}\d{2}[A-Z0-9]{11})", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b")
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")

TABLE_3_1_ROW_LABELS = {
    "outward_taxable_supplies": r"outward taxable supplies\s*\(other than",
    "outward_zero_rated": r"outward taxable supplies\s*\(zero rated\)",
    "other_outward_nil_exempt": r"other outward supplies\s*\(nil",
    "inward_reverse_charge": r"inward supplies.*reverse charge",
    "non_gst_outward": r"non-?gst outward supplies",
}


def _clean_number(s):
    """Parse a GST-PDF-formatted number string ('1,23,456.00', '-', '') to float."""
    if s is None:
        return 0.0
    s = str(s).strip().replace(",", "")
    if s in ("", "-", "—", "NA", "N/A"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        match = NUMBER_PATTERN.search(s)
        return float(match.group().replace(",", "")) if match else 0.0


def extract_gstr1_fields(pdf_path):
    """
    Extract structured fields from a GSTR-1 summary PDF.

    Inputs:
        pdf_path: str - path to the PDF file

    Returns:
        dict with:
            "gstin": str or None
            "return_period": str or None
            "arn": str or None
            "filing_date": str or None (as found in the PDF, 'DD/MM/YYYY')
            "table_3_1": dict[row_key] -> {"taxable_value", "igst", "cgst",
                "sgst", "cess"} (all float)
            "table_12_hsn": list[dict] - one row per HSN entry (raw values)
            "table_13_documents": list[dict] - one row per document type
            "raw_text": str - full extracted text, for debugging/fallback

    Raises:
        ValueError - if the PDF has no extractable text layer (likely
            scanned/raster — this parser only handles text-layer PDFs)
    """
    full_text = []
    tables_by_page = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            full_text.append(text)
            tables_by_page.append(page.extract_tables())
    raw_text = "\n".join(full_text)

    if not raw_text.strip():
        raise ValueError(
            "No extractable text found in this PDF — it may be a scanned "
            "image rather than a text-layer PDF. This parser doesn't do OCR."
        )

    gstin_match = GSTIN_PATTERN.search(raw_text)
    arn_match = ARN_PATTERN.search(raw_text)
    dates = DATE_PATTERN.findall(raw_text)

    period_match = re.search(r"(?:Period|Tax [Pp]eriod)[:\s]+([A-Za-z]+[\s-]+\d{4})", raw_text)

    result = {
        "gstin": gstin_match.group() if gstin_match else None,
        "return_period": period_match.group(1) if period_match else None,
        "arn": arn_match.group(1) if arn_match else None,
        "filing_date": dates[-1] if dates else None,  # filing date is typically the last date on the form
        "table_3_1": _extract_table_31(raw_text, tables_by_page),
        "table_12_hsn": _extract_table_12(tables_by_page),
        "table_13_documents": _extract_table_13(tables_by_page),
        "raw_text": raw_text,
    }
    return result


def _extract_table_31(raw_text, tables_by_page):
    """
    Extract Table 3.1 (outward supplies summary) — tries structured table
    extraction first, falls back to line-by-line regex on raw text since
    real-world PDF table extraction is often imperfect for bordered forms.
    """
    rows = {}

    # Attempt 1: structured tables (pdfplumber)
    for page_tables in tables_by_page:
        for table in page_tables:
            for row in table:
                if not row or row[0] is None:
                    continue
                row_text = str(row[0]).lower().replace("\n", " ")
                for key, pattern in TABLE_3_1_ROW_LABELS.items():
                    if re.search(pattern, row_text) and key not in rows:
                        nums = [_clean_number(c) for c in row[1:] if c is not None]
                        if len(nums) >= 4:
                            rows[key] = {
                                "taxable_value": nums[0], "igst": nums[1],
                                "cgst": nums[2], "sgst": nums[3],
                                "cess": nums[4] if len(nums) > 4 else 0.0,
                            }

    # Attempt 2: line-based regex fallback for any rows structured extraction missed
    if len(rows) < len(TABLE_3_1_ROW_LABELS):
        for line in raw_text.split("\n"):
            line_lower = line.lower()
            for key, pattern in TABLE_3_1_ROW_LABELS.items():
                if key in rows:
                    continue
                if re.search(pattern, line_lower):
                    nums = [_clean_number(n) for n in NUMBER_PATTERN.findall(line)]
                    if len(nums) >= 4:
                        rows[key] = {
                            "taxable_value": nums[0], "igst": nums[1],
                            "cgst": nums[2], "sgst": nums[3],
                            "cess": nums[4] if len(nums) > 4 else 0.0,
                        }

    return rows


def _extract_table_12(tables_by_page):
    """Extract Table 12 (HSN-wise summary) rows from structured tables."""
    hsn_rows = []
    for page_tables in tables_by_page:
        for table in page_tables:
            header = [str(c).lower() if c else "" for c in (table[0] if table else [])]
            if not any("hsn" in h for h in header):
                continue
            for row in table[1:]:
                if not row or row[0] is None or not re.match(r"^\d+$", str(row[0]).strip()):
                    continue
                nums = [_clean_number(c) for c in row]
                hsn_rows.append({
                    "hsn_code": row[1] if len(row) > 1 else None,
                    "total_quantity": nums[4] if len(nums) > 4 else 0.0,
                    "total_value": nums[5] if len(nums) > 5 else 0.0,
                    "total_taxable_value": nums[6] if len(nums) > 6 else 0.0,
                    "total_tax": sum(nums[7:10]) if len(nums) > 9 else 0.0,
                })
    return hsn_rows


def _extract_table_13(tables_by_page):
    """Extract Table 13 (documents issued) rows from structured tables."""
    doc_rows = []
    for page_tables in tables_by_page:
        for table in page_tables:
            header = [str(c).lower() if c else "" for c in (table[0] if table else [])]
            if not any("document" in h or "invoice" in h for h in header):
                continue
            for row in table[1:]:
                if not row or row[0] is None:
                    continue
                nums = [_clean_number(c) for c in row]
                if not nums or all(n == 0 for n in nums):
                    continue
                doc_rows.append({
                    "nature_of_document": row[0],
                    "total_number": nums[-2] if len(nums) >= 2 else (nums[-1] if nums else 0.0),
                    "cancelled": nums[-1] if len(nums) >= 2 else 0.0,
                })
    return doc_rows
