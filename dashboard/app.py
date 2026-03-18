import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import sqlite3
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.preprocess import preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DB_PATH = os.path.join("database", "student_dropout.db")
MODEL_PATH = os.path.join("models", "dropout_model.pkl")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DropoutIQ · Student Risk Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL STYLES ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #07080f;
    --bg2:       #0d0f1c;
    --surface:   rgba(255,255,255,0.04);
    --border:    rgba(255,255,255,0.08);
    --neon:      #00f5c4;
    --neon2:     #7c5cfc;
    --neon3:     #ff4d6d;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
}

/* ── Base Reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-head) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { font-family: var(--font-head) !important; }

/* Sidebar radio buttons */
div[role="radiogroup"] label {
    padding: 10px 18px !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    font-size: 0.9rem !important;
    color: var(--muted) !important;
}
div[role="radiogroup"] label:hover {
    background: var(--surface) !important;
    color: var(--text) !important;
}
div[role="radiogroup"] label[data-checked="true"],
div[role="radiogroup"] label[aria-checked="true"] {
    background: linear-gradient(135deg, rgba(0,245,196,0.15), rgba(124,92,252,0.15)) !important;
    color: var(--neon) !important;
    border: 1px solid rgba(0,245,196,0.3) !important;
}

/* ── Metric Cards ── */
div[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.4rem 1.6rem !important;
    position: relative !important;
    overflow: hidden !important;
    backdrop-filter: blur(12px) !important;
}
div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--neon), var(--neon2));
}
div[data-testid="metric-container"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-head) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: var(--neon) !important;
}

/* ── Buttons ── */
button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, var(--neon), var(--neon2)) !important;
    color: #07080f !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.65rem 2rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Sliders ── */
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--neon) !important;
}

/* ── Inputs / Selects ── */
input, select, textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font-head) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 14px; overflow: hidden; }
.stDataFrame table { font-family: var(--font-mono) !important; font-size: 0.82rem !important; }

/* ── Alert/success/error banners ── */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border-width: 1px !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}

/* ── Matplotlib figure backgrounds ── */
.stPlotlyChart, .stImage { border-radius: 14px !important; }

/* ── Subheader tweaks ── */
h1 { font-family: var(--font-head) !important; font-weight: 800 !important; font-size: 2rem !important; }
h2, h3 { font-family: var(--font-head) !important; font-weight: 700 !important; }

/* ── Section divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
    margin: 2rem 0 !important;
}

/* ── Number input ── */
div[data-testid="stNumberInput"] input { font-family: var(--font-mono) !important; }
</style>
""", unsafe_allow_html=True)


# ─── MATPLOTLIB DARK THEME ──────────────────────────────────────────────────────
def set_chart_style():
    mpl.rcParams.update({
        "figure.facecolor":  "#0d0f1c",
        "axes.facecolor":    "#0d0f1c",
        "axes.edgecolor":    "#1f2133",
        "axes.labelcolor":   "#9ca3af",
        "axes.titlecolor":   "#e8eaf0",
        "xtick.color":       "#6b7280",
        "ytick.color":       "#6b7280",
        "text.color":        "#e8eaf0",
        "grid.color":        "#1f2133",
        "grid.linewidth":    0.6,
        "axes.grid":         True,
        "font.family":       "monospace",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

NEON_PALETTE = ["#00f5c4", "#7c5cfc", "#ff4d6d", "#f5a623", "#4fc3f7", "#b2ff59"]


# ─── LOAD DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM student_records;", conn)
    conn.close()
    return df

df = load_data()
model = joblib.load(MODEL_PATH)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 2rem; text-align: center;'>
        <div style='font-size:2.6rem; margin-bottom:4px;'>🧠</div>
        <div style='font-family:"Syne",sans-serif; font-weight:800; font-size:1.15rem; color:#e8eaf0; letter-spacing:0.04em;'>DropoutIQ</div>
        <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#6b7280; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px;'>Student Risk Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        ["Overview", "EDA", "Model Performance", "Prediction", "Add Student Data"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#374151; text-align:center; padding-top:1rem; border-top:1px solid #1f2133; letter-spacing:0.08em;'>
        {len(df):,} RECORDS LOADED
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown("""
    <div style='margin-bottom:0.4rem;'>
        <span style='font-family:"Space Mono",monospace; font-size:0.72rem; color:#6b7280; letter-spacing:0.14em; text-transform:uppercase;'>DASHBOARD</span>
    </div>
    <h1 style='margin-top:0; background:linear-gradient(135deg,#00f5c4,#7c5cfc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.1;'>Student Dropout<br>Risk Intelligence</h1>
    <p style='color:#6b7280; font-size:0.92rem; margin-bottom:2.5rem; max-width:520px;'>Real-time monitoring and predictive analytics for academic retention across streams and cohorts.</p>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    dropout_rate  = df["dropped_out"].mean() * 100
    active_count  = (df["dropped_out"] == 0).sum()
    dropout_count = (df["dropped_out"] == 1).sum()

    col1.metric("Total Students",  f"{len(df):,}")
    col2.metric("Dropout Rate",    f"{dropout_rate:.1f}%")
    col3.metric("Avg CGPA",        f"{df['cgpa'].mean():.2f}")
    col4.metric("Active Students", f"{active_count:,}")

    st.markdown("<hr>", unsafe_allow_html=True)

    set_chart_style()
    col_a, col_b = st.columns([1.6, 1])

    with col_a:
        st.markdown("#### Dropout Rate by Stream")
        data = df.groupby("stream")["dropped_out"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.barh(data.index, data.values * 100, color=NEON_PALETTE[:len(data)], edgecolor="none", height=0.6)
        ax.set_xlabel("Dropout Rate (%)")
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        for bar, val in zip(bars, data.values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{val*100:.1f}%", va="center", fontsize=8, color="#9ca3af")
        fig.tight_layout()
        st.pyplot(fig)

    with col_b:
        st.markdown("#### Enrollment Status")
        fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
        wedges, texts, autotexts = ax2.pie(
            [active_count, dropout_count],
            labels=["Active", "Dropped"],
            autopct="%1.1f%%",
            colors=["#00f5c4", "#ff4d6d"],
            startangle=90,
            wedgeprops=dict(edgecolor="#07080f", linewidth=3),
        )
        for t in autotexts: t.set_color("#07080f"); t.set_fontweight("bold")
        for t in texts: t.set_color("#9ca3af")
        fig2.patch.set_facecolor("#0d0f1c")
        fig2.tight_layout()
        st.pyplot(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":

    st.markdown("""
    <span style='font-family:"Space Mono",monospace; font-size:0.72rem; color:#6b7280; letter-spacing:0.14em; text-transform:uppercase;'>ANALYSIS</span>
    <h1 style='margin-top:0; color:#e8eaf0;'>Exploratory Data Analysis</h1>
    """, unsafe_allow_html=True)

    set_chart_style()

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Dropout Rate by Stream")
        data = df.groupby("stream")["dropped_out"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        bars = ax.bar(data.index, data.values * 100, color=NEON_PALETTE[:len(data)], edgecolor="none", width=0.6)
        ax.set_ylabel("Dropout Rate (%)")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### Dropout Rate by Year")
        data2 = df.groupby("academic_year")["dropped_out"].mean()
        fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
        ax2.plot(data2.index, data2.values * 100, color="#00f5c4", linewidth=2.5, marker="o",
                 markersize=6, markerfacecolor="#07080f", markeredgewidth=2)
        ax2.fill_between(data2.index, data2.values * 100, alpha=0.12, color="#00f5c4")
        ax2.set_ylabel("Dropout Rate (%)")
        ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        fig2.tight_layout()
        st.pyplot(fig2)

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### CGPA Distribution by Status")
        dropped = df[df["dropped_out"] == 1]["cgpa"]
        active  = df[df["dropped_out"] == 0]["cgpa"]
        fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
        ax3.hist(active,   bins=25, alpha=0.7, color="#00f5c4", label="Active",  edgecolor="none")
        ax3.hist(dropped,  bins=25, alpha=0.7, color="#ff4d6d", label="Dropped", edgecolor="none")
        ax3.set_xlabel("CGPA")
        ax3.legend(framealpha=0, labelcolor="white")
        fig3.tight_layout()
        st.pyplot(fig3)

    with col4:
        st.markdown("#### Attendance Distribution by Status")
        dropped_att = df[df["dropped_out"] == 1]["attendance"]
        active_att  = df[df["dropped_out"] == 0]["attendance"]
        fig4, ax4 = plt.subplots(figsize=(5.5, 3.5))
        ax4.hist(active_att,  bins=25, alpha=0.7, color="#7c5cfc", label="Active",  edgecolor="none")
        ax4.hist(dropped_att, bins=25, alpha=0.7, color="#ff4d6d", label="Dropped", edgecolor="none")
        ax4.set_xlabel("Attendance (%)")
        ax4.legend(framealpha=0, labelcolor="white")
        fig4.tight_layout()
        st.pyplot(fig4)

    # Row 3 — correlation heatmap
    st.markdown("#### Feature Correlation Heatmap")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_cols].corr()
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("neon", ["#ff4d6d", "#0d0f1c", "#00f5c4"])
    im = ax5.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax5.set_xticks(range(len(num_cols))); ax5.set_xticklabels(num_cols, rotation=30, ha="right", fontsize=8)
    ax5.set_yticks(range(len(num_cols))); ax5.set_yticklabels(num_cols, fontsize=8)
    plt.colorbar(im, ax=ax5, fraction=0.03, pad=0.02)
    fig5.tight_layout()
    st.pyplot(fig5)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":

    st.markdown("""
    <span style='font-family:"Space Mono",monospace; font-size:0.72rem; color:#6b7280; letter-spacing:0.14em; text-transform:uppercase;'>EVALUATION</span>
    <h1 style='margin-top:0; color:#e8eaf0;'>Model Performance</h1>
    """, unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(),
        "Decision Tree":       DecisionTreeClassifier()
    }

    results = []
    with st.spinner("Training models…"):
        for name, m in models.items():
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            results.append({
                "Model":     name,
                "Accuracy":  accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall":    recall_score(y_test, pred),
                "F1 Score":  f1_score(y_test, pred),
            })

    df_results = pd.DataFrame(results)
    best_idx   = df_results["Accuracy"].idxmax()
    best_model = df_results.loc[best_idx, "Model"]

    # Metric cards for each model
    cols = st.columns(3)
    for i, row in df_results.iterrows():
        with cols[i]:
            is_best = row["Model"] == best_model
            border  = "rgba(0,245,196,0.5)" if is_best else "rgba(255,255,255,0.08)"
            badge   = "<span style='font-size:0.65rem; background:#00f5c4; color:#07080f; border-radius:6px; padding:2px 8px; font-weight:700; margin-left:8px;'>BEST</span>" if is_best else ""
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.04); border:1px solid {border}; border-radius:16px; padding:1.4rem 1.6rem; position:relative;'>
                <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#6b7280; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;'>{row["Model"]}{badge}</div>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'>
                    <div><div style='font-size:0.65rem; color:#6b7280;'>Accuracy</div><div style='font-size:1.35rem; font-weight:800; color:#00f5c4;'>{row["Accuracy"]:.3f}</div></div>
                    <div><div style='font-size:0.65rem; color:#6b7280;'>F1 Score</div><div style='font-size:1.35rem; font-weight:800; color:#7c5cfc;'>{row["F1 Score"]:.3f}</div></div>
                    <div><div style='font-size:0.65rem; color:#6b7280;'>Precision</div><div style='font-size:1.1rem; font-weight:700; color:#e8eaf0;'>{row["Precision"]:.3f}</div></div>
                    <div><div style='font-size:0.65rem; color:#6b7280;'>Recall</div><div style='font-size:1.1rem; font-weight:700; color:#e8eaf0;'>{row["Recall"]:.3f}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Grouped bar chart
    set_chart_style()
    st.markdown("#### Metric Comparison")
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = range(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (_, row) in enumerate(df_results.iterrows()):
        offset = (i - 1) * width
        vals = [row[m] for m in metrics]
        ax.bar([xi + offset for xi in x], vals, width=width * 0.9,
               color=NEON_PALETTE[i], edgecolor="none", label=row["Model"])
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.legend(framealpha=0, labelcolor="white", loc="upper right")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":

    st.markdown("""
    <span style='font-family:"Space Mono",monospace; font-size:0.72rem; color:#6b7280; letter-spacing:0.14em; text-transform:uppercase;'>INFERENCE</span>
    <h1 style='margin-top:0; color:#e8eaf0;'>Predict Dropout Risk</h1>
    <p style='color:#6b7280; font-size:0.9rem; margin-bottom:2rem;'>Fill in student details to generate an AI-powered risk assessment.</p>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:18px; padding:1.6rem 1.8rem; margin-bottom:1.5rem;'>
        <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#6b7280; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:1rem;'>Academic Profile</div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            age    = st.number_input("Age",    17, 30, 20)
            gender = st.selectbox("Gender", ["Male", "Female"])
            stream = st.selectbox("Stream", ["CSE","ECE","Mechanical","Civil","BBA","BCom"])
        with col2:
            attendance = st.slider("Attendance %", 0, 100, 75)
            cgpa       = st.slider("CGPA", 0.0, 10.0, 7.0)
            fees       = st.selectbox("Fees Paid %", [60, 75, 100])
        with col3:
            hostel     = st.selectbox("Hostel", [0, 1], format_func=lambda x: "Yes" if x else "No")
            distance   = st.slider("Distance (km)", 0, 50, 10)
            scholarship= st.selectbox("Scholarship", [0, 1], format_func=lambda x: "Yes" if x else "No")
            year       = st.number_input("Academic Year", 2020, 2030, 2024)

        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⚡  Run Risk Assessment"):

        input_df = pd.DataFrame([[
            age,
            1 if gender == "Male" else 0,
            ["CSE","ECE","Mechanical","Civil","BBA","BCom"].index(stream),
            attendance,
            cgpa,
            fees,
            hostel,
            distance,
            scholarship,
            year,
        ]], columns=[
            "age","gender","stream","attendance","cgpa",
            "fees_paid","hostel","distance_km","scholarship","academic_year",
        ])

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        pct  = int(prob * 100)
        color = "#ff4d6d" if pred == 1 else "#00f5c4"
        label = "HIGH RISK" if pred == 1 else "LOW RISK"
        icon  = "⚠️" if pred == 1 else "✅"

        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03); border:1px solid {color}40; border-radius:18px; padding:2rem 2.4rem; margin-top:1rem; display:flex; align-items:center; gap:2rem;'>
            <div style='font-size:3.5rem; line-height:1;'>{icon}</div>
            <div style='flex:1;'>
                <div style='font-family:"Space Mono",monospace; font-size:0.7rem; color:#6b7280; letter-spacing:0.14em; text-transform:uppercase;'>Risk Assessment</div>
                <div style='font-size:2rem; font-weight:800; color:{color}; letter-spacing:0.04em; margin:4px 0;'>{label}</div>
                <div style='font-size:0.9rem; color:#9ca3af;'>Dropout probability: <strong style='color:{color};'>{pct}%</strong></div>
            </div>
            <div style='text-align:right;'>
                <div style='font-size:3.5rem; font-weight:800; color:{color}; font-family:"Space Mono",monospace; line-height:1;'>{pct}<span style='font-size:1.5rem;'>%</span></div>
                <div style='font-size:0.65rem; color:#6b7280; letter-spacing:0.1em;'>CONFIDENCE</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk gauge bar
        fill_color = color
        st.markdown(f"""
        <div style='margin-top:1.2rem;'>
            <div style='height:8px; background:rgba(255,255,255,0.07); border-radius:99px; overflow:hidden;'>
                <div style='height:100%; width:{pct}%; background:linear-gradient(90deg,#7c5cfc,{fill_color}); border-radius:99px; transition:width 0.6s ease;'></div>
            </div>
            <div style='display:flex; justify-content:space-between; font-family:"Space Mono",monospace; font-size:0.6rem; color:#4b5563; margin-top:4px;'>
                <span>0%</span><span>50%</span><span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ADD STUDENT DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Add Student Data":

    st.markdown("""
    <span style='font-family:"Space Mono",monospace; font-size:0.72rem; color:#6b7280; letter-spacing:0.14em; text-transform:uppercase;'>DATA ENTRY</span>
    <h1 style='margin-top:0; color:#e8eaf0;'>Add Student Record</h1>
    <p style='color:#6b7280; font-size:0.9rem; margin-bottom:2rem;'>Insert a new student record into the training database.</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:18px; padding:1.6rem 1.8rem;'>
    <div style='font-family:"Space Mono",monospace; font-size:0.65rem; color:#6b7280; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:1rem;'>Student Details</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age    = st.number_input("Age",    17, 30, 20,  key="add_age")
        gender = st.selectbox("Gender", ["Male", "Female"],         key="add_gender")
        stream = st.selectbox("Stream", ["CSE","ECE","Mechanical","Civil","BBA","BCom"], key="add_stream")
        year   = st.number_input("Academic Year", 2020, 2035, 2026, key="add_year")
    with col2:
        attendance = st.slider("Attendance %", 0, 100, 75, key="add_att")
        cgpa       = st.slider("CGPA", 0.0, 10.0, 7.0,    key="add_cgpa")
        fees       = st.selectbox("Fees %", [60, 75, 100], key="add_fees")
    with col3:
        hostel      = st.selectbox("Hostel",  [0,1], format_func=lambda x: "Yes" if x else "No", key="add_hostel")
        distance    = st.slider("Distance (km)", 0, 50, 10, key="add_dist")
        scholarship = st.selectbox("Scholarship", [0,1], format_func=lambda x: "Yes" if x else "No", key="add_schol")
        dropped     = st.selectbox("Dropped Out", [0,1], format_func=lambda x: "Yes" if x else "No", key="add_drop")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("💾  Insert Record"):
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO student_records
            (age, gender, stream, attendance, cgpa, fees_paid, hostel, distance_km, scholarship, dropped_out, academic_year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (age, gender, stream, attendance, cgpa, fees, hostel, distance, scholarship, dropped, year))
        conn.commit()
        conn.close()
        st.success("✅  Record inserted successfully. Cache cleared — refresh to see updated counts.")
        st.cache_data.clear()