"""
app.py
------
A clean, light, SaaS-style Streamlit dashboard for the spam predictor —
restyled after a "product dashboard" look (white cards, sidebar nav,
black/blue accents) instead of the old dark theme. Every text-producing
element has an explicit color so nothing ever renders invisible.

Layout:
  - Left sidebar: brand + navigation (Overview, Single Check, Batch
    Upload, History) — mirrors a typical SaaS dashboard sidebar.
  - Overview: hero + model performance stat cards.
  - Single Check: paste/upload/example text, get a verdict, a
    confidence gauge, and influential words.
  - Batch Upload: upload a .csv/.txt of many messages, get a results
    table, spam/ham breakdown chart, and a downloadable CSV.
  - History: everything checked this session.

Run with:  streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Shield | Email Spam Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Design tokens (light SaaS-dashboard palette)
# ------------------------------------------------------------------
INK = "#0F172A"       # primary text
SUBTLE = "#64748B"    # secondary / caption text
LINE = "#E2E8F0"      # borders
CARD = "#FFFFFF"      # card background
CANVAS = "#F7F8FA"    # page background
BLUE = "#2563EB"      # accent / links / info
GREEN = "#16A34A"     # ham / safe
RED = "#DC2626"       # spam / danger
AMBER = "#D97706"     # warning

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp {{
        font-family: 'Inter', -apple-system, sans-serif;
        background: {CANVAS};
    }}

    /* Force readable dark text everywhere by default (fixes invisible-text issues) */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div, .stApp li {{
        color: {INK};
    }}

    #MainMenu, footer {{ visibility: hidden; }}

    /* ---------------- Sidebar ---------------- */
    section[data-testid="stSidebar"] {{
        background: {CARD};
        border-right: 1px solid {LINE};
    }}
    section[data-testid="stSidebar"] * {{
        color: {INK} !important;
    }}
    .brand {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0 1.2rem 0;
        border-bottom: 1px solid {LINE};
        margin-bottom: 1rem;
    }}
    .brand .logo-dot {{
        width: 30px; height: 30px;
        border-radius: 8px;
        background: linear-gradient(135deg, {BLUE}, #60A5FA);
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
    }}
    .brand-name {{
        font-weight: 800;
        font-size: 1.15rem;
        color: {INK} !important;
    }}
    section[data-testid="stSidebar"] .stRadio > label {{
        color: {SUBTLE} !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {{
        padding: 0.55rem 0.7rem;
        border-radius: 10px;
        margin-bottom: 2px;
    }}
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {{
        background: {CANVAS};
    }}
    .sidebar-caption {{
        color: {SUBTLE} !important;
        font-size: 0.78rem;
        line-height: 1.4;
    }}

    /* ---------------- Hero ---------------- */
    .hero-badge {{
        display: inline-block;
        background: {CANVAS};
        border: 1px solid {LINE};
        color: {SUBTLE} !important;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }}
    .hero h1 {{
        font-size: 2.4rem;
        font-weight: 800;
        color: {INK} !important;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }}
    .hero p {{
        color: {SUBTLE} !important;
        font-size: 1.05rem;
        max-width: 640px;
    }}

    /* ---------------- Cards ---------------- */
    .panel {{
        background: {CARD};
        border: 1px solid {LINE};
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .panel h3, .panel h4 {{
        color: {INK} !important;
        margin-top: 0;
    }}

    .stat-card {{
        background: {CARD};
        border: 1px solid {LINE};
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .stat-card .stat-label {{
        color: {SUBTLE} !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }}
    .stat-card .stat-value {{
        color: {INK} !important;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }}
    .stat-card .stat-delta {{
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.15rem;
    }}
    .delta-up {{ color: {GREEN} !important; }}
    .delta-flat {{ color: {SUBTLE} !important; }}

    /* ---------------- Text area ---------------- */
    .stTextArea textarea {{
        background-color: {CANVAS} !important;
        color: {INK} !important;
        border: 1px solid {LINE} !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }}
    .stTextArea textarea::placeholder {{
        color: #94A3B8 !important;
        opacity: 1 !important;
    }}
    .stTextArea label {{ color: {SUBTLE} !important; }}

    /* ---------------- Buttons ---------------- */
    .stButton button {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        border: 1px solid {LINE} !important;
    }}
    .stButton button[kind="primary"] {{
        background: {INK} !important;
        color: #FFFFFF !important;
        border: 1px solid {INK} !important;
    }}
    .stButton button[kind="primary"] p {{ color: #FFFFFF !important; }}
    .stDownloadButton button {{
        background: {BLUE} !important;
        color: #FFFFFF !important;
        border: 1px solid {BLUE} !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }}
    .stDownloadButton button p {{ color: #FFFFFF !important; }}

    /* ---------------- Result banners ---------------- */
    .result-spam {{
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
    }}
    .result-ham {{
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
    }}
    .result-spam h2 {{ color: {RED} !important; font-size: 1.4rem; margin: 0; letter-spacing: 0.02em; }}
    .result-ham h2 {{ color: {GREEN} !important; font-size: 1.4rem; margin: 0; letter-spacing: 0.02em; }}
    .result-spam p, .result-ham p {{ color: {SUBTLE} !important; margin-top: 0.3rem; }}

    /* ---------------- Word chips ---------------- */
    .chip {{
        display: inline-block;
        padding: 0.25rem 0.7rem;
        margin: 0.2rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .chip-spam {{ background: #FEF2F2; color: {RED} !important; border: 1px solid #FCA5A5; }}
    .chip-ham {{ background: #F0FDF4; color: {GREEN} !important; border: 1px solid #86EFAC; }}

    .section-caption {{ color: {SUBTLE} !important; font-size: 0.85rem; }}

    /* ---------------- Tabs (kept for sub-navigation inside pages) ---------------- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {CANVAS};
        border-radius: 8px 8px 0 0;
        color: {SUBTLE} !important;
        padding: 0.5rem 1.1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {CARD} !important;
        color: {INK} !important;
        border: 1px solid {LINE};
        border-bottom: none;
    }}

    /* ---------------- File uploader ---------------- */
    [data-testid="stFileUploader"] section {{
        background-color: {CANVAS} !important;
        border: 1px dashed #CBD5E1 !important;
        border-radius: 10px !important;
    }}
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] section small {{
        color: {SUBTLE} !important;
    }}

    /* ---------------- Badges ---------------- */
    .badge-row {{ display: flex; gap: 0.6rem; margin-top: 0.6rem; }}
    .badge {{
        flex: 1;
        text-align: center;
        border-radius: 10px;
        padding: 0.6rem;
        font-weight: 700;
        font-size: 1.05rem;
        border: 1px solid {LINE};
    }}
    .badge-total {{ background: #EFF6FF; color: {BLUE} !important; border-color: #BFDBFE; }}
    .badge-spam {{ background: #FEF2F2; color: {RED} !important; border-color: #FCA5A5; }}
    .badge-ham {{ background: #F0FDF4; color: {GREEN} !important; border-color: #86EFAC; }}

    /* ---------------- History rows ---------------- */
    .history-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0.2rem;
        border-bottom: 1px solid {LINE};
    }}
    .history-text {{ color: {INK} !important; font-size: 0.9rem; }}
    .tag-spam {{ color: {RED} !important; font-weight: 700; font-size: 0.8rem; }}
    .tag-safe {{ color: {GREEN} !important; font-weight: 700; font-size: 0.8rem; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load model, preprocessor, and training metrics
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not (os.path.exists("spam_model.pkl") and os.path.exists("spam_preprocessor.pkl")):
        return None, None
    model = joblib.load("spam_model.pkl")
    preprocessor = joblib.load("spam_preprocessor.pkl")
    return model, preprocessor


def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            return json.load(f)
    return None


model, preprocessor = load_artifacts()
metrics = load_metrics()

if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "Overview"


def predict_single(text):
    """Runs one message through the model. Returns (label, confidence%, spam_prob%, vec)."""
    vec = preprocessor.transform([text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    confidence = probability[prediction] * 100
    spam_prob = probability[1] * 100
    label = "SPAM" if prediction == 1 else "NOT SPAM"
    return label, confidence, spam_prob, vec


def extract_text_column(df):
    """Reuses the same column-name detection as dataset.py so batch
    uploads accept the same variety of formats (v1/v2, label/text,
    Category/Message, email/label, etc.)."""
    cols = [c.strip().lower() for c in df.columns]
    df = df.copy()
    df.columns = cols

    for candidate in ("text", "message", "v2", "email", "content", "body"):
        if candidate in cols:
            return df[candidate].astype(str)

    return df[cols[0]].astype(str)


def gauge_chart(spam_prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=spam_prob,
        number={"suffix": "%", "font": {"color": INK, "size": 28}},
        title={"text": "Spam Probability", "font": {"color": SUBTLE, "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": SUBTLE, "tickfont": {"color": SUBTLE}},
            "bar": {"color": BLUE},
            "bgcolor": CANVAS,
            "bordercolor": LINE,
            "steps": [
                {"range": [0, 40], "color": "rgba(22,163,74,0.15)"},
                {"range": [40, 70], "color": "rgba(217,119,6,0.15)"},
                {"range": [70, 100], "color": "rgba(220,38,38,0.15)"},
            ],
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
    )
    return fig


def bar_chart(ham_count, spam_count):
    fig = go.Figure(data=[go.Bar(
        x=["Not Spam", "Spam"],
        y=[ham_count, spam_count],
        marker_color=[GREEN, RED],
        text=[ham_count, spam_count],
        textposition="outside",
    )])
    fig.update_layout(
        height=260,
        margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        xaxis=dict(gridcolor=LINE, color=INK),
        yaxis=dict(gridcolor=LINE, color=INK),
    )
    return fig

# ------------------------------------------------------------------
# Sidebar — brand + navigation
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="logo-dot">🛡️</div>
        <div class="brand-name">Spam Shield</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Overview", "Single Check", "Batch Upload", "History"],
        index=["Overview", "Single Check", "Batch Upload", "History"].index(st.session_state.page),
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#E2E8F0;">', unsafe_allow_html=True)
    if metrics:
        st.markdown(
            f"<p class='sidebar-caption'><b>Model</b><br>{metrics['model_type']}<br>"
            f"Trained {metrics['trained_at']}</p>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<p class='sidebar-caption'>AI-powered spam &amp; ham classification "
        "for email and SMS text.</p>",
        unsafe_allow_html=True,
    )

if model is None:
    st.error(
        "No trained model found. Run `python train.py` first — this creates "
        "spam_model.pkl, spam_preprocessor.pkl, and metrics.json."
    )
    st.stop()

# ====================================================================
# PAGE: Overview
# ====================================================================
if st.session_state.page == "Overview":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Real-time classification</div>
        <h1>Catch spam before it reaches the inbox</h1>
        <p>Spam Shield scores any email or SMS message in real time and shows you
        exactly which words drove the verdict — check one message at a time or
        run a full batch from a file.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        stat_data = [
            (c1, "Accuracy", f"{metrics['cv_accuracy_mean']*100:.1f}%", f"± {metrics['cv_accuracy_std']*100:.1f}"),
            (c2, "Precision", f"{metrics['cv_precision_mean']*100:.1f}%", f"± {metrics['cv_precision_std']*100:.1f}"),
            (c3, "Recall", f"{metrics['cv_recall_mean']*100:.1f}%", f"± {metrics['cv_recall_std']*100:.1f}"),
            (c4, "F1 Score", f"{metrics['cv_f1_mean']*100:.1f}%", f"± {metrics['cv_f1_std']*100:.1f}"),
            (c5, "Training Samples", f"{metrics['total_samples']:,}", "5-fold CV"),
        ]
        for col, label, value, delta in stat_data:
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                    <div class="stat-delta delta-flat">{delta}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(
            f"<p class='section-caption' style='margin-top:0.8rem;'>"
            f"All headline metrics come from 5-fold cross-validation for consistency "
            f"across accuracy, precision, recall, and F1.</p>",
            unsafe_allow_html=True,
        )
        if metrics["total_samples"] < 200:
            st.markdown(
                "<p class='section-caption'>Note: this model was trained on a small "
                "dataset. Metrics will become more stable with more training data.</p>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("No metrics.json found — run train.py to generate performance stats.")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Check a single message")
        st.markdown(
            "<p class='section-caption'>Paste or upload one message and see an "
            "instant verdict with a confidence gauge.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Go to Single Check →", use_container_width=True):
            st.session_state.page = "Single Check"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Check many at once")
        st.markdown(
            "<p class='section-caption'>Upload a CSV or TXT file of messages and "
            "download the predictions as a spreadsheet.</p>",
            unsafe_allow_html=True,
        )
        if st.button("Go to Batch Upload →", use_container_width=True):
            st.session_state.page = "Batch Upload"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ====================================================================
# PAGE: Single Check
# ====================================================================
elif st.session_state.page == "Single Check":
    st.markdown("## Single Message Check")
    st.markdown(
        "<p class='section-caption'>Paste text directly, upload a .txt file, or "
        "load a quick example.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1.1, 1])

    with col_input:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Enter a message")

        uploaded_txt = st.file_uploader(
            "Upload a .txt file (optional)", type=["txt"], key="single_txt_upload"
        )
        if uploaded_txt is not None and st.session_state.get("_last_upload") != uploaded_txt.file_id:
            st.session_state["message_box"] = uploaded_txt.read().decode("utf-8", errors="ignore")
            st.session_state["_last_upload"] = uploaded_txt.file_id

        message = st.text_area(
            "Message text",
            height=180,
            placeholder="e.g. Congratulations! You've won a free prize, click here to claim...",
            label_visibility="collapsed",
            key="message_box",
        )

        st.markdown(
            "<p class='section-caption'>Load a pre-written sample to see how the "
            "model reacts to an obvious spam message versus a normal one.</p>",
            unsafe_allow_html=True,
        )
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            if st.button("Load spam example", use_container_width=True):
                st.session_state["message_box"] = (
                    "URGENT! You've won a $1000 Walmart gift card. Click here now "
                    "to claim your free prize before it expires!"
                )
                st.rerun()
        with example_col2:
            if st.button("Load normal example", use_container_width=True):
                st.session_state["message_box"] = (
                    "Hey, just checking if we're still on for the meeting "
                    "tomorrow at 10am. Let me know!"
                )
                st.rerun()

        check_clicked = st.button("Check Message", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Result")

        if check_clicked and message.strip():
            label, confidence, spam_prob, vec = predict_single(message)

            st.session_state.history.append({
                "text": message,
                "label": label,
                "confidence": confidence,
            })

            if label == "SPAM":
                st.markdown(f"""
                <div class="result-spam">
                    <h2>SPAM DETECTED</h2>
                    <p>{confidence:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-ham">
                    <h2>LOOKS SAFE</h2>
                    <p>{confidence:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)

            st.plotly_chart(gauge_chart(spam_prob), use_container_width=True)

            try:
                feature_names = preprocessor.vectorizer.get_feature_names_out()
                words_in_msg = vec.nonzero()[1]
                spam_scores = (
                    model.feature_log_prob_[1][words_in_msg]
                    - model.feature_log_prob_[0][words_in_msg]
                )
                word_scores = sorted(
                    zip([feature_names[i] for i in words_in_msg], spam_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                if word_scores:
                    st.markdown(
                        "<p class='section-caption'><b>Words that influenced this result:</b></p>",
                        unsafe_allow_html=True,
                    )
                    chips_html = ""
                    for word, score in word_scores[:8]:
                        css_class = "chip-spam" if score > 0 else "chip-ham"
                        chips_html += f'<span class="chip {css_class}">{word}</span>'
                    st.markdown(chips_html, unsafe_allow_html=True)
            except Exception:
                pass

        elif check_clicked:
            st.warning("Please enter a message first.")
        else:
            st.markdown(
                "<p class='section-caption' style='text-align:center;'>"
                "Enter a message and click <b>Check Message</b> to see the "
                "prediction here.</p>",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

# ====================================================================
# PAGE: Batch Upload
# ====================================================================
elif st.session_state.page == "Batch Upload":
    st.markdown("## Batch Upload")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### Check many messages at once")
    st.markdown(
        "<p class='section-caption'>Upload a <b>.csv</b> file (with a text/message "
        "column — 'text', 'message', 'v2', 'email', etc.) or a <b>.txt</b> file with "
        "one message per line. Every message gets a prediction, shown below in a "
        "downloadable table.</p>",
        unsafe_allow_html=True,
    )

    batch_file = st.file_uploader(
        "Upload a .csv or .txt file", type=["csv", "txt"], key="batch_upload"
    )

    if batch_file is not None:
        try:
            if batch_file.name.endswith(".csv"):
                raw_df = pd.read_csv(batch_file, encoding="latin-1")
                texts = extract_text_column(raw_df)
            else:
                content = batch_file.read().decode("utf-8", errors="ignore")
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                texts = pd.Series(lines)

            if len(texts) == 0:
                st.warning("No messages found in the uploaded file.")
            else:
                results = []
                for text in texts:
                    label, confidence, spam_prob, _ = predict_single(text)
                    results.append({
                        "Message": text[:120] + ("..." if len(text) > 120 else ""),
                        "Prediction": label,
                        "Confidence (%)": round(confidence, 1),
                    })
                results_df = pd.DataFrame(results)

                spam_count = int((results_df["Prediction"] == "SPAM").sum())
                ham_count = int((results_df["Prediction"] == "NOT SPAM").sum())

                st.markdown(f"""
                <div class="badge-row">
                    <div class="badge badge-total">{len(results_df)} checked</div>
                    <div class="badge badge-spam">{spam_count} spam</div>
                    <div class="badge badge-ham">{ham_count} not spam</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(results_df, use_container_width=True, height=350)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name="spam_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.plotly_chart(bar_chart(ham_count, spam_count), use_container_width=True)

        except Exception as e:
            st.error(f"Couldn't process this file: {e}")
    else:
        st.markdown(
            "<p class='section-caption' style='text-align:center;'>"
            "Upload a file above to check multiple messages at once.</p>",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ====================================================================
# PAGE: History
# ====================================================================
elif st.session_state.page == "History":
    st.markdown("## Session History")
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            "<p class='section-caption' style='text-align:center;'>"
            "Nothing checked yet this session — head to Single Check or Batch "
            "Upload to get started.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<p class='section-caption'>{len(st.session_state.history)} message(s) "
            f"checked this session.</p>",
            unsafe_allow_html=True,
        )
        for item in reversed(st.session_state.history[-25:]):
            tag_class = "tag-spam" if item["label"] == "SPAM" else "tag-safe"
            tag_text = "SPAM" if item["label"] == "SPAM" else "SAFE"
            st.markdown(f"""
            <div class="history-row">
                <span class="history-text">{item['text'][:80]}</span>
                <span class="{tag_class}">{tag_text} · {item['confidence']:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)