"""
Streamlit app — two ways to check GST compliance risk:

  Tab 1: Upload a CSV in the trained model's schema (Column0..Column21) ->
         scored by the GNN + Isolation Forest pipeline.
  Tab 2: Upload a real GSTR-1 PDF -> checked by rule-based anomaly checks
         on the real, named GSTR-1 fields (tax reconciliation, filing
         delay, HSN mismatch, etc).

These are deliberately SEPARATE pipelines — see src/gstr1_pdf/parser.py
docstring for why GSTR-1 fields aren't fed into the anonymized-column GNN.

Run with:
    streamlit run app/streamlit_app.py

Tab 1 requires models/ to already contain a trained model — run
`python run_pipeline.py` first (see README.md). Tab 2 has no such
requirement — it's a standalone rule engine.
"""

import os
import sys
import tempfile
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.phase3_pipeline.model_pipeline import GSTComplianceRiskModel
from src.gstr1_pdf.parser import extract_gstr1_fields
from src.gstr1_pdf.anomaly_rules import run_all_checks

st.set_page_config(page_title="GST Compliance Risk Checker", page_icon="🧾", layout="wide")


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    return GSTComplianceRiskModel.load()


def render_csv_tab():
    st.caption(
        "Upload a CSV of GST taxpayer records (in the trained model's anonymized "
        "feature schema) to score them for compliance risk. Scoring combines a Graph "
        "Neural Network (trained on taxpayer similarity) with an Isolation Forest "
        "anomaly detector — new records are attached inductively to the trained "
        "model's reference graph, no retraining needed."
    )

    if not os.path.exists(config.GNN_WEIGHTS_PATH):
        st.error(
            "No trained model found in `models/`. Run `python run_pipeline.py` "
            "from the project root first, then restart this app."
        )
        return

    model = load_model()

    with st.expander("Expected file format", expanded=False):
        st.markdown(
            f"CSV with these columns (an `{config.ID_COL}` column is optional but "
            f"recommended — rows are auto-numbered if missing):"
        )
        st.code(", ".join([config.ID_COL] + config.NUMERIC_COLS), language=None)
        st.caption(
            "Column names follow the anonymized GST-Predictive-Dataset-2024 schema "
            "(Column0..Column21). Missing values are fine — they're imputed the "
            "same way the training data was. If you have a real GSTR-1 filing "
            "instead of this schema, use the 'GSTR-1 PDF' tab."
        )

    uploaded_file = st.file_uploader("Upload GST taxpayer CSV", type=["csv"], key="csv_uploader")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        explain = st.checkbox("Explain top risk drivers", value=False,
                               help="Slower — runs a per-row graph explanation. "
                                    "Recommended only for smaller uploads (<200 rows).")

    if uploaded_file is None:
        st.info("Upload a CSV to get started, or try the sample data below.")
        if st.button("Use a small real sample instead"):
            uploaded_file = config.TEST_FEATURES_PATH

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file, nrows=2000)
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")
            return

        missing_cols = [c for c in config.NUMERIC_COLS if c not in raw_df.columns]
        if missing_cols:
            st.error(f"Uploaded file is missing required columns: {missing_cols}")
            return

        st.success(f"Loaded {len(raw_df)} rows.")
        if len(raw_df) > 500 and explain:
            st.warning(
                f"{len(raw_df)} rows with explanations on — this may take a while. "
                f"Consider unchecking 'Explain top risk drivers' for large uploads."
            )

        with st.spinner("Scoring rows..."):
            results = model.predict_for_new_data(raw_df, explain=explain)

        st.subheader("Results")

        tier_counts = results["risk_tier"].value_counts()
        cols = st.columns(4)
        for col, tier in zip(cols, ["Compliant", "Low Risk", "Elevated Risk", "High Risk"]):
            col.metric(tier, int(tier_counts.get(tier, 0)))

        def highlight_risk(row):
            color = {
                "High Risk": "background-color: #ffcccc",
                "Elevated Risk": "background-color: #ffe0b3",
                "Low Risk": "background-color: #fff9cc",
                "Compliant": "background-color: #d9f2d9",
            }.get(row["risk_tier"], "")
            return [color] * len(row)

        display_df = results.sort_values("final_risk_score", ascending=False)
        st.dataframe(display_df.style.apply(highlight_risk, axis=1), use_container_width=True)

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", csv_bytes,
                            file_name="gst_risk_scores.csv", mime="text/csv")

        st.subheader("Risk score distribution")
        st.bar_chart(results["final_risk_score"])

        if explain:
            st.subheader("Top risk drivers (highest-scored rows)")
            top_risky = display_df.head(5)
            for _, row in top_risky.iterrows():
                with st.expander(f"{row[config.ID_COL]} — {row['risk_tier']} "
                                  f"(score: {row['final_risk_score']:.3f})"):
                    st.write(f"GNN risk probability: {row['gnn_prob']:.3f}")
                    st.write(f"Anomaly score: {row['anomaly_score']:.3f}")
                    st.write(f"Top contributing features: {', '.join(row['top_features'])}")


def render_gstr1_pdf_tab():
    st.caption(
        "Upload a real GSTR-1 summary PDF (as downloaded from the GST portal). "
        "This runs independent, rule-based checks on the actual filing figures — "
        "tax reconciliation, filing delay, HSN-wise consistency, cancelled-invoice "
        "rate — and explains exactly why anything looks off. This does **not** use "
        "the GNN model (its features are anonymized survey columns with no known "
        "correspondence to real GSTR-1 fields — see project README for why)."
    )

    with st.expander("What gets checked", expanded=False):
        st.markdown(
            "- **Tax reconciliation** — does declared IGST+CGST+SGST match the "
            "taxable value at a standard GST rate slab (5/12/18/28%)?\n"
            "- **Zero tax on taxable supplies** — taxable supplies declared with "
            "near-zero tax paid\n"
            "- **Filing delay** — filed vs. the 11th-of-next-month due date\n"
            "- **Cancelled documents** — unusually high invoice cancellation rate\n"
            "- **HSN reconciliation** — does the HSN-wise summary (Table 12) sum "
            "match the Table 3.1 total?"
        )
        st.caption(
            "Built and tested against the standard public GSTR-1 PDF layout. Real "
            "filings from different portal versions or filing software may have "
            "layout variations — if extraction looks wrong, the raw extracted text "
            "is shown below for debugging."
        )

    uploaded_pdf = st.file_uploader("Upload GSTR-1 PDF", type=["pdf"], key="pdf_uploader")

    if uploaded_pdf is None:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Extracting fields from PDF..."):
            fields = extract_gstr1_fields(tmp_path)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Could not parse this PDF: {e}")
        return
    finally:
        os.unlink(tmp_path)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("GSTIN", fields["gstin"] or "Not found")
    col2.metric("Return period", fields["return_period"] or "Not found")
    col3.metric("Filing date", fields["filing_date"] or "Not found")
    col4.metric("ARN", fields["arn"] or "Not found")

    if not fields["table_3_1"]:
        st.warning(
            "Could not extract Table 3.1 (outward supplies summary) from this PDF — "
            "checks below will be limited. This usually means the PDF layout "
            "differs from the standard format this parser targets."
        )

    result = run_all_checks(fields)

    st.subheader("Verdict")
    if result["is_anomalous"]:
        st.error(f"⚠️ Flagged as anomalous — anomaly score: {result['anomaly_score']:.2f} / 1.00")
    else:
        st.success(f"✅ No anomalies detected — anomaly score: {result['anomaly_score']:.2f} / 1.00")

    st.subheader("Why")
    for check in result["all_checks"]:
        icon = "🔴" if check["triggered"] else "🟢"
        st.markdown(f"{icon} **{check['check']}**: {check['message']}")

    with st.expander("Raw extracted text (for debugging extraction issues)", expanded=False):
        st.text(fields["raw_text"][:5000])


def main():
    st.title("🧾 GST Compliance Risk Checker")

    tab1, tab2 = st.tabs(["📄 GSTR-1 PDF (rule-based)", "📊 CSV (trained model)"])
    with tab1:
        render_gstr1_pdf_tab()
    with tab2:
        render_csv_tab()


if __name__ == "__main__":
    main()