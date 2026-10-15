"""
GSTR-1 rule-based anomaly detector.

Operates on the fields extracted by gstr1_pdf/parser.py — real GSTR-1
figures (taxable value, IGST/CGST/SGST, invoice counts, HSN summary), not
the anonymized model features. Each check is a plain arithmetic or
statistical rule with a human-readable explanation, chosen specifically
so a real filer or auditor could understand and verify WHY something was
flagged — unlike the GNN's SHAP/embedding-based explanations, which are
only meaningful in terms of anonymized columns.

Standard GST rate slabs (5/12/18/28%) are used for the tax-reconciliation
check — these are the standard slabs as of the dataset/document's era;
update RATE_SLABS if slabs change or if a filer is under a special/
composition scheme this checker doesn't account for.
"""

import numpy as np

RATE_SLABS = [0.0, 0.05, 0.12, 0.18, 0.28]
TAX_RECONCILIATION_TOLERANCE = 0.02   # 2% relative tolerance on tax-vs-rate check
GSTR1_DUE_DAY_OF_MONTH = 11            # standard monthly GSTR-1 due date
LATE_FILING_FLAG_DAYS = 10             # days past due date to flag as an anomaly
ROUND_NUMBER_SHARE_THRESHOLD = 0.6     # share of invoices that are suspiciously round
CANCELLED_RATIO_THRESHOLD = 0.15       # share of documents cancelled


def check_tax_reconciliation(table_3_1):
    """
    Check whether declared tax (IGST + CGST + SGST) on outward taxable
    supplies reconciles with the declared taxable value at ANY standard
    GST rate slab, within tolerance.

    Inputs:
        table_3_1: dict - output of parser._extract_table_31()

    Returns:
        (triggered: bool, message: str, severity: float in [0,1])
    """
    row = table_3_1.get("outward_taxable_supplies")
    if row is None or row["taxable_value"] == 0:
        return False, "No outward taxable supplies declared — reconciliation check skipped.", 0.0

    taxable_value = row["taxable_value"]
    declared_tax = row["igst"] + row["cgst"] + row["sgst"]
    implied_rate = declared_tax / taxable_value if taxable_value else 0.0

    closest_slab = min(RATE_SLABS, key=lambda r: abs(r - implied_rate))
    relative_error = abs(implied_rate - closest_slab) / max(closest_slab, 1e-6) if closest_slab > 0 else implied_rate

    if closest_slab == 0.0 and implied_rate > TAX_RECONCILIATION_TOLERANCE:
        return True, (
            f"Declared tax (₹{declared_tax:,.0f}) implies a {implied_rate:.1%} rate on "
            f"₹{taxable_value:,.0f} taxable value, which doesn't match any standard GST "
            f"slab (0/5/12/18/28%)."
        ), min(1.0, implied_rate / 0.28)

    if relative_error > TAX_RECONCILIATION_TOLERANCE:
        return True, (
            f"Declared tax (₹{declared_tax:,.0f}) implies a {implied_rate:.1%} effective rate on "
            f"₹{taxable_value:,.0f} taxable value — doesn't reconcile with the nearest standard "
            f"slab ({closest_slab:.0%}) within tolerance. Possible misreporting or mixed-rate "
            f"supplies not broken out separately."
        ), min(1.0, relative_error)

    return False, f"Tax reconciles with the {closest_slab:.0%} slab within tolerance.", 0.0


def check_zero_tax_on_taxable_supplies(table_3_1):
    """
    Flag taxable supplies declared with essentially zero tax paid — a
    classic misclassification or evasion pattern (e.g. wrongly claiming
    nil-rated/exempt treatment for taxable supplies).
    """
    row = table_3_1.get("outward_taxable_supplies")
    if row is None:
        return False, "No outward taxable supplies row found.", 0.0

    taxable_value = row["taxable_value"]
    total_tax = row["igst"] + row["cgst"] + row["sgst"]
    if taxable_value > 10000 and total_tax < taxable_value * 0.005:
        return True, (
            f"₹{taxable_value:,.0f} in taxable supplies declared with only ₹{total_tax:,.0f} "
            f"tax paid (<0.5% effective rate) — taxable supplies should attract a standard "
            f"GST rate; this may indicate misclassification as nil-rated/exempt."
        ), 0.8
    return False, "Tax paid is consistent with taxable-supply treatment.", 0.0


def check_filing_delay(filing_date_str, return_period_str):
    """
    Flag late filing relative to the standard GSTR-1 due date (11th of
    the month following the return period).

    Inputs:
        filing_date_str: str or None - 'DD/MM/YYYY' as extracted from the PDF
        return_period_str: str or None - e.g. 'June-2024'

    Returns:
        (triggered, message, severity)
    """
    if not filing_date_str or not return_period_str:
        return False, "Filing date or return period not found in PDF — delay check skipped.", 0.0

    from datetime import datetime
    import calendar

    try:
        filing_date = datetime.strptime(filing_date_str.replace("-", "/"), "%d/%m/%Y")
        period_date = datetime.strptime(return_period_str.strip(), "%B-%Y")
    except ValueError:
        return False, "Could not parse filing date / return period — delay check skipped.", 0.0

    # Due date = 11th of the month AFTER the return period
    if period_date.month == 12:
        due_year, due_month = period_date.year + 1, 1
    else:
        due_year, due_month = period_date.year, period_date.month + 1
    due_date = datetime(due_year, due_month, GSTR1_DUE_DAY_OF_MONTH)

    delay_days = (filing_date - due_date).days
    if delay_days > LATE_FILING_FLAG_DAYS:
        return True, (
            f"Filed {delay_days} days after the {due_date.strftime('%d %b %Y')} due date "
            f"for the {return_period_str} return period."
        ), min(1.0, delay_days / 60)
    return False, f"Filed on time (within {LATE_FILING_FLAG_DAYS} days of due date).", 0.0


def check_cancelled_documents(table_13_documents):
    """
    Flag an unusually high share of cancelled invoices/documents — can
    indicate invoice manipulation (issuing then cancelling to inflate or
    manage reported figures).
    """
    total_docs = sum(row["total_number"] for row in table_13_documents)
    total_cancelled = sum(row["cancelled"] for row in table_13_documents)
    if total_docs == 0:
        return False, "No documents declared — cancellation check skipped.", 0.0

    ratio = total_cancelled / total_docs
    if ratio > CANCELLED_RATIO_THRESHOLD:
        return True, (
            f"{total_cancelled:.0f} of {total_docs:.0f} documents ({ratio:.1%}) were cancelled — "
            f"well above the typical range, which can indicate invoice manipulation."
        ), min(1.0, ratio)
    return False, f"Cancellation rate ({ratio:.1%}) is within a typical range.", 0.0


def check_hsn_reconciliation(table_3_1, table_12_hsn):
    """
    Flag a mismatch between Table 3.1's total taxable value and the sum
    of Table 12's HSN-wise taxable values — they should reconcile.
    """
    row = table_3_1.get("outward_taxable_supplies")
    if row is None or not table_12_hsn:
        return False, "Insufficient data for HSN reconciliation check.", 0.0

    table_31_total = row["taxable_value"]
    hsn_total = sum(r["total_taxable_value"] for r in table_12_hsn)
    if table_31_total == 0:
        return False, "No taxable value declared — HSN reconciliation check skipped.", 0.0

    relative_diff = abs(table_31_total - hsn_total) / table_31_total
    if relative_diff > 0.05:
        return True, (
            f"Table 3.1 declares ₹{table_31_total:,.0f} in taxable value, but the HSN-wise "
            f"summary (Table 12) sums to ₹{hsn_total:,.0f} — a {relative_diff:.1%} mismatch. "
            f"These should reconcile; a gap can indicate incomplete or inconsistent reporting."
        ), min(1.0, relative_diff)
    return False, "HSN-wise summary reconciles with Table 3.1 total.", 0.0


def run_all_checks(gstr1_fields):
    """
    Run every anomaly check against a parsed GSTR-1 field dict and
    combine into an overall flagged/not-flagged verdict.

    Inputs:
        gstr1_fields: dict - output of parser.extract_gstr1_fields()

    Returns:
        dict with:
            "is_anomalous": bool
            "anomaly_score": float in [0,1] - max severity across triggered checks
            "triggered_reasons": list[str] - human-readable explanations
            "all_checks": list[dict] - every check's result, triggered or not
    """
    table_3_1 = gstr1_fields.get("table_3_1", {})
    table_12 = gstr1_fields.get("table_12_hsn", [])
    table_13 = gstr1_fields.get("table_13_documents", [])

    checks = [
        ("Tax reconciliation", check_tax_reconciliation(table_3_1)),
        ("Zero tax on taxable supplies", check_zero_tax_on_taxable_supplies(table_3_1)),
        ("Filing delay", check_filing_delay(gstr1_fields.get("filing_date"),
                                             gstr1_fields.get("return_period"))),
        ("Cancelled documents", check_cancelled_documents(table_13)),
        ("HSN reconciliation", check_hsn_reconciliation(table_3_1, table_12)),
    ]

    all_checks = [
        {"check": name, "triggered": triggered, "message": message, "severity": severity}
        for name, (triggered, message, severity) in checks
    ]
    triggered_checks = [c for c in all_checks if c["triggered"]]

    anomaly_score = max((c["severity"] for c in triggered_checks), default=0.0)

    return {
        "is_anomalous": len(triggered_checks) > 0,
        "anomaly_score": round(anomaly_score, 3),
        "triggered_reasons": [c["message"] for c in triggered_checks],
        "all_checks": all_checks,
    }
