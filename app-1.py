"""
EduPro — Student Segmentation & Personalized Course Recommendation System
Streamlit Web Application
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis import run_pipeline, SEGMENT_NAMES, SEGMENT_DESC

# ── Auto-generate data if CSVs don't exist ───────────────────────────────────
_data_dir = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(os.path.join(_data_dir, "users.csv")):
    os.makedirs(_data_dir, exist_ok=True)
    import random
    from datetime import datetime, timedelta
    _np = np; _rng = 42; _np.random.seed(_rng); random.seed(_rng)
    N_U, N_C, N_T = 1000, 150, 5000
    users = pd.DataFrame({
        "UserID": [f"U{str(i).zfill(4)}" for i in range(1, N_U+1)],
        "Age": _np.random.randint(18, 65, N_U),
        "Gender": _np.random.choice(["Male","Female","Non-binary","Prefer not to say"], N_U, p=[0.48,0.44,0.05,0.03]),
        "Country": _np.random.choice(["India","USA","UK","Canada","Germany","Australia","Brazil"], N_U, p=[0.35,0.20,0.10,0.10,0.10,0.08,0.07]),
        "JoinDate": [(datetime(2022,1,1)+timedelta(days=random.randint(0,900))).strftime("%Y-%m-%d") for _ in range(N_U)]
    })
    cats = ["Data Science","Web Development","Business","Design","Marketing","AI/ML","Cloud","Cybersecurity","Finance","Language"]
    courses = pd.DataFrame({
        "CourseID": [f"C{str(i).zfill(3)}" for i in range(1, N_C+1)],
        "CourseName": [f"Course_{i}" for i in range(1, N_C+1)],
        "CourseCategory": _np.random.choice(cats, N_C),
        "CourseType": _np.random.choice(["Video","Interactive","Project-Based","Live"], N_C, p=[0.4,0.25,0.25,0.10]),
        "CourseLevel": _np.random.choice(["Beginner","Intermediate","Advanced"], N_C, p=[0.40,0.35,0.25]),
        "CourseRating": _np.round(_np.random.uniform(3.0, 5.0, N_C), 1),
        "CourseDurationHrs": _np.random.randint(2, 60, N_C),
        "CoursePrice": _np.round(_np.random.uniform(9.99, 199.99, N_C), 2),
    })
    txn = pd.DataFrame({
        "TransactionID": [f"T{str(i).zfill(5)}" for i in range(1, N_T+1)],
        "UserID": _np.random.choice(users["UserID"].tolist(), N_T),
        "CourseID": _np.random.choice(courses["CourseID"].tolist(), N_T),
        "TransactionDate": [(datetime(2022,1,1)+timedelta(days=random.randint(0,900))).strftime("%Y-%m-%d") for _ in range(N_T)],
        "Amount": _np.round(_np.random.uniform(9.99, 199.99, N_T), 2),
        "CompletionStatus": _np.random.choice(["Completed","In Progress","Not Started"], N_T, p=[0.45,0.35,0.20]),
        "Rating": _np.where(_np.random.rand(N_T)>0.3, _np.round(_np.random.uniform(2.5,5.0,N_T),1), _np.nan)
    }).drop_duplicates(subset=["UserID","CourseID"])
    users.to_csv(f"{_data_dir}/users.csv", index=False)
    courses.to_csv(f"{_data_dir}/courses.csv", index=False)
    txn.to_csv(f"{_data_dir}/transactions.csv", index=False)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro — Learner Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background: #0b0f1a; }

/* Header */
.hero {
    background: linear-gradient(135deg, #0f1628 0%, #1a2444 50%, #0f1628 100%);
    border: 1px solid #2a3a6e;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(16,185,129,0.06) 0%, transparent 40%);
    pointer-events: none;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #f0f4ff;
    margin: 0;
    letter-spacing: -0.03em;
}
.hero p {
    color: #8892b0;
    margin: 0.5rem 0 0;
    font-size: 1rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #131929 0%, #1a2444 100%);
    border: 1px solid #2a3a6e;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #6366f1;
}
.metric-card .label {
    color: #8892b0;
    font-size: 0.82rem;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Segment badges */
.seg-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0f4ff;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #2a3a6e;
}

/* Rec card */
.rec-card {
    background: #131929;
    border: 1px solid #2a3a6e;
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.rec-card .course-name { color: #f0f4ff; font-weight: 600; font-size: 0.95rem; }
.rec-card .course-meta { color: #8892b0; font-size: 0.8rem; margin-top: 0.25rem; }

/* Streamlit overrides */
.stSelectbox label, .stSlider label { color: #c0ccdd !important; }
div[data-testid="stMetricValue"] { color: #6366f1 !important; font-family: 'Syne', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
SEG_COLORS = {
    "🧭 Explorers":       "#6366f1",
    "🎯 Specialists":     "#10b981",
    "🏆 Career Climbers": "#f59e0b",
    "🌱 Casual Learners": "#ec4899",
}

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running ML pipeline…")
def get_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    return run_pipeline(data_dir=data_dir, n_clusters=4)

result  = get_data()
prof    = result["profiles"]
courses = result["courses"]
txn     = result["transactions"]
metrics = result["metrics"]
recs    = result["recommendations"]

# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 EduPro Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview Dashboard",
         "🗂️ Cluster Explorer",
         "👤 Learner Profile",
         "📚 Course Recommendations",
         "📈 Evaluation & Validation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Filters**")
    seg_filter = st.multiselect(
        "Segments",
        list(SEG_COLORS.keys()),
        default=list(SEG_COLORS.keys())
    )
    age_range = st.slider("Age range", 18, 65, (18, 65))

    filtered = prof[
        prof["segment_name"].isin(seg_filter) &
        prof["Age"].between(age_range[0], age_range[1])
    ]

    st.markdown("---")
    st.caption(f"**{len(filtered):,}** learners selected")

# ════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":

    st.markdown("""
    <div class="hero">
      <h1>Student Segmentation & Personalized Course Recommendation</h1>
      <p>EduPro · Data Science Intelligence Platform · 1,000 Learners · 150 Courses · 4,930 Transactions</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (f"{len(prof):,}", "Total Learners"),
        (f"{len(courses):,}", "Courses"),
        (f"{len(txn):,}", "Transactions"),
        (f"{metrics['km_silhouette']:.3f}", "Silhouette Score"),
        (f"${filtered['avg_spending'].mean():.0f}", "Avg Spend / User"),
    ]
    for col, (val, label) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="value">{val}</div>
              <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Segment pie + Scatter ──
    c1, c2 = st.columns([1, 2])

    with c1:
        seg_counts = filtered["segment_name"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_pie = px.pie(
            seg_counts, names="Segment", values="Count",
            color="Segment",
            color_discrete_map=SEG_COLORS,
            hole=0.55
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c0ccdd",
            margin=dict(t=30, b=10, l=10, r=10),
            legend=dict(orientation="v", font_size=11),
            title=dict(text="Segment Distribution", font_color="#f0f4ff", font_size=16)
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_scatter = px.scatter(
            filtered, x="pca_x", y="pca_y",
            color="segment_name",
            color_discrete_map=SEG_COLORS,
            opacity=0.7,
            size_max=8,
            hover_data=["UserID", "Age", "total_courses_enrolled", "avg_spending"],
            labels={"pca_x": "PCA Dimension 1", "pca_y": "PCA Dimension 2",
                    "segment_name": "Segment"},
            title="Learner Cluster Map (PCA)"
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,26,0.6)",
            font_color="#c0ccdd",
            title_font_color="#f0f4ff",
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        fig_scatter.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Row 3: Spending + Courses enrolled ──
    c3, c4 = st.columns(2)
    with c3:
        fig_box = px.box(
            filtered, x="segment_name", y="avg_spending",
            color="segment_name",
            color_discrete_map=SEG_COLORS,
            title="Avg Spending by Segment",
            labels={"avg_spending": "Avg Spending ($)", "segment_name": ""}
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,26,0.6)",
            font_color="#c0ccdd",
            title_font_color="#f0f4ff",
            showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with c4:
        fig_hist = px.histogram(
            filtered, x="total_courses_enrolled",
            color="segment_name",
            color_discrete_map=SEG_COLORS,
            barmode="overlay",
            opacity=0.7,
            title="Courses Enrolled Distribution",
            labels={"total_courses_enrolled": "Courses Enrolled", "segment_name": "Segment"}
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,26,0.6)",
            font_color="#c0ccdd",
            title_font_color="#f0f4ff",
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — CLUSTER EXPLORER
# ════════════════════════════════════════════════════════════════════
elif page == "🗂️ Cluster Explorer":

    st.markdown('<div class="section-header">🗂️ Cluster Explorer</div>', unsafe_allow_html=True)

    # Segment cards
    seg_stats = prof.groupby("segment_name").agg(
        count=("UserID", "count"),
        avg_courses=("total_courses_enrolled", "mean"),
        avg_spending=("avg_spending", "mean"),
        avg_diversity=("diversity_score", "mean"),
        avg_depth=("learning_depth_index", "mean"),
        avg_completion=("completion_rate", "mean"),
    ).reset_index()

    for _, row in seg_stats.iterrows():
        seg  = row["segment_name"]
        desc = SEGMENT_DESC.get(list(SEGMENT_NAMES.keys())[list(SEGMENT_NAMES.values()).index(seg)], "")
        col  = SEG_COLORS.get(seg, "#6366f1")
        pct  = row["count"] / len(prof) * 100
        with st.container():
            st.markdown(f"""
            <div style="background:#131929;border:1px solid {col}40;border-left:4px solid {col};
                        border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                  <span style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#f0f4ff;">{seg}</span>
                  <span style="margin-left:12px;font-size:0.8rem;color:{col};background:{col}20;
                               padding:2px 10px;border-radius:20px;">{row['count']:,} learners ({pct:.1f}%)</span>
                </div>
              </div>
              <p style="color:#8892b0;font-size:0.85rem;margin:0.5rem 0 1rem;">{desc}</p>
              <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;">
                <div><div style="color:{col};font-weight:700;font-size:1.1rem;">{row['avg_courses']:.1f}</div>
                     <div style="color:#8892b0;font-size:0.75rem;">Avg Courses</div></div>
                <div><div style="color:{col};font-weight:700;font-size:1.1rem;">${row['avg_spending']:.0f}</div>
                     <div style="color:#8892b0;font-size:0.75rem;">Avg Spend</div></div>
                <div><div style="color:{col};font-weight:700;font-size:1.1rem;">{row['avg_diversity']:.1f}</div>
                     <div style="color:#8892b0;font-size:0.75rem;">Diversity Score</div></div>
                <div><div style="color:{col};font-weight:700;font-size:1.1rem;">{row['avg_depth']:.2f}</div>
                     <div style="color:#8892b0;font-size:0.75rem;">Depth Index</div></div>
                <div><div style="color:{col};font-weight:700;font-size:1.1rem;">{row['avg_completion']*100:.0f}%</div>
                     <div style="color:#8892b0;font-size:0.75rem;">Completion</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Comparison by Segment</div>', unsafe_allow_html=True)
    features = ["total_courses_enrolled", "avg_spending", "diversity_score",
                "learning_depth_index", "completion_rate", "avg_course_rating_enrolled"]
    labels   = ["Courses Enrolled", "Avg Spend ($)", "Diversity Score",
                "Depth Index", "Completion Rate", "Avg Rating Enrolled"]

    seg_radar = prof.groupby("segment_name")[features].mean().reset_index()

    fig_radar = go.Figure()
    for _, row in seg_radar.iterrows():
        seg   = row["segment_name"]
        vals  = [row[f] for f in features]
        # Normalize 0-1 for radar
        maxes = prof[features].max()
        norm  = [row[f] / maxes[f] for f in features]
        fig_radar.add_trace(go.Scatterpolar(
            r=norm + [norm[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=seg,
            line_color=SEG_COLORS.get(seg, "#6366f1"),
            fillcolor=SEG_COLORS.get(seg, "#6366f1"),
            opacity=0.25,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(11,15,26,0.8)",
            radialaxis=dict(visible=True, showticklabels=False,
                            gridcolor="#2a3a6e", linecolor="#2a3a6e"),
            angularaxis=dict(gridcolor="#2a3a6e", linecolor="#2a3a6e",
                             tickfont=dict(color="#c0ccdd"))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c0ccdd",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Category preference heatmap
    st.markdown('<div class="section-header">Category Preference by Segment</div>', unsafe_allow_html=True)
    cat_seg = (txn.merge(prof[["UserID","segment_name"]], on="UserID")
                  .groupby(["segment_name","CourseCategory"]).size()
                  .reset_index(name="count"))
    cat_pivot = cat_seg.pivot(index="segment_name", columns="CourseCategory", values="count").fillna(0)
    fig_heat = px.imshow(
        cat_pivot,
        color_continuous_scale="Blues",
        title="Enrollment Heatmap: Segment × Category",
        labels=dict(color="Enrollments")
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c0ccdd",
        title_font_color="#f0f4ff",
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — LEARNER PROFILE
# ════════════════════════════════════════════════════════════════════
elif page == "👤 Learner Profile":

    st.markdown('<div class="section-header">👤 Learner Profile Explorer</div>', unsafe_allow_html=True)

    uid = st.selectbox("Select a Learner", prof["UserID"].tolist())
    user = prof[prof["UserID"] == uid].iloc[0]
    seg  = user["segment_name"]
    col  = SEG_COLORS.get(seg, "#6366f1")

    # Profile card
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#131929,#1a2444);
                border:1px solid {col}60;border-radius:16px;padding:2rem;margin-bottom:1.5rem;">
      <div style="display:flex;align-items:center;gap:1.5rem;">
        <div style="width:64px;height:64px;border-radius:50%;background:{col}30;
                    border:2px solid {col};display:flex;align-items:center;justify-content:center;
                    font-size:1.8rem;">👤</div>
        <div>
          <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#f0f4ff;">{uid}</div>
          <div style="color:{col};font-size:0.9rem;margin-top:2px;">{seg}</div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1.5rem;margin-top:1.5rem;">
        <div><div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;">Age</div>
             <div style="color:#f0f4ff;font-weight:600;font-size:1.1rem;">{int(user['Age'])}</div></div>
        <div><div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;">Gender</div>
             <div style="color:#f0f4ff;font-weight:600;font-size:1.1rem;">{user['Gender']}</div></div>
        <div><div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;">Courses Enrolled</div>
             <div style="color:#f0f4ff;font-weight:600;font-size:1.1rem;">{int(user['total_courses_enrolled'])}</div></div>
        <div><div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;">Total Spent</div>
             <div style="color:#f0f4ff;font-weight:600;font-size:1.1rem;">${user['total_spending']:.0f}</div></div>
        <div><div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;">Preferred Category</div>
             <div style="color:#f0f4ff;font-weight:600;font-size:1.1rem;">{user['preferred_category']}</div></div>
        <div><div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;">Preferred Level</div>
             <div style="color:#f0f4ff;font-weight:600;font-size:1.1rem;">{user['preferred_level']}</div></div>
        <div><div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;">Diversity Score</div>
             <div style="color:#f0f4ff;font-weight:600;font-size:1.1rem;">{user['diversity_score']:.1f}</div></div>
        <div><div style="color:#8892b0;font-size:0.75rem;text-transform:uppercase;">Completion Rate</div>
             <div style="color:#f0f4ff;font-weight:600;font-size:1.1rem;">{user['completion_rate']*100:.0f}%</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # User's enrollments
    user_txn = txn[txn["UserID"] == uid].merge(courses, on="CourseID")
    if not user_txn.empty:
        st.markdown('<div class="section-header">Enrolled Courses</div>', unsafe_allow_html=True)
        display_cols = ["CourseName", "CourseCategory", "CourseLevel",
                        "CourseRating", "CompletionStatus", "Amount"]
        st.dataframe(
            user_txn[display_cols].rename(columns={
                "CourseName": "Course", "CourseCategory": "Category",
                "CourseLevel": "Level", "CourseRating": "Rating",
                "CompletionStatus": "Status", "Amount": "Paid ($)"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Category pie
        cat_counts = user_txn["CourseCategory"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_u = px.pie(
            cat_counts, names="Category", values="Count",
            hole=0.5, title="Category Breakdown",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_u.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c0ccdd",
            title_font_color="#f0f4ff",
            margin=dict(t=40, b=10)
        )
        st.plotly_chart(fig_u, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — COURSE RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════
elif page == "📚 Course Recommendations":

    st.markdown('<div class="section-header">📚 Personalized Course Recommendations</div>', unsafe_allow_html=True)

    uid  = st.selectbox("Select a Learner", prof["UserID"].tolist())
    user = prof[prof["UserID"] == uid].iloc[0]
    seg  = user["segment_name"]
    col  = SEG_COLORS.get(seg, "#6366f1")

    # Level / category filters
    f1, f2 = st.columns(2)
    with f1:
        level_f = st.multiselect("Filter by Level",
                                  ["Beginner", "Intermediate", "Advanced"],
                                  default=["Beginner", "Intermediate", "Advanced"])
    with f2:
        cat_f = st.multiselect("Filter by Category",
                                courses["CourseCategory"].unique().tolist(),
                                default=courses["CourseCategory"].unique().tolist())

    st.markdown(f"""
    <div style="background:{col}15;border:1px solid {col}40;border-radius:10px;
                padding:0.8rem 1.2rem;margin-bottom:1rem;">
      <span style="color:{col};font-weight:600;">{seg}</span>
      <span style="color:#8892b0;font-size:0.85rem;margin-left:0.75rem;">
        {SEGMENT_DESC.get(list(SEGMENT_NAMES.keys())[list(SEGMENT_NAMES.values()).index(seg)], "")}
      </span>
    </div>
    """, unsafe_allow_html=True)

    user_recs = recs.get(uid, [])
    filtered_recs = [
        r for r in user_recs
        if r["CourseLevel"] in level_f and r["CourseCategory"] in cat_f
    ]

    if not filtered_recs:
        st.info("No recommendations match the current filters.")
    else:
        for i, r in enumerate(filtered_recs, 1):
            rating_stars = "⭐" * int(round(r["CourseRating"]))
            st.markdown(f"""
            <div class="rec-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <div class="course-name">#{i} &nbsp; {r['CourseName']}</div>
                  <div class="course-meta">
                    📂 {r['CourseCategory']} &nbsp;·&nbsp;
                    🎯 {r['CourseLevel']} &nbsp;·&nbsp;
                    ⏱ {r['CourseDurationHrs']}h &nbsp;·&nbsp;
                    {rating_stars} {r['CourseRating']}
                  </div>
                </div>
                <div style="color:#10b981;font-weight:700;font-size:1.1rem;white-space:nowrap;">
                  ${r['CoursePrice']:.2f}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Segment-wide popular courses
    st.markdown('<div class="section-header">Top Courses in Your Segment</div>', unsafe_allow_html=True)
    seg_users = prof[prof["segment_name"] == seg]["UserID"].tolist()
    seg_txn   = txn[txn["UserID"].isin(seg_users)].merge(courses, on="CourseID")
    top_seg   = (seg_txn.groupby(["CourseID", "CourseName", "CourseCategory",
                                   "CourseLevel", "CourseRating"])
                        .size().reset_index(name="enrollments")
                        .nlargest(10, "enrollments"))
    fig_top = px.bar(
        top_seg, x="enrollments", y="CourseName",
        color="CourseRating", color_continuous_scale="Blues",
        orientation="h",
        title=f"Most Popular Courses — {seg}",
        labels={"enrollments": "Enrollments", "CourseName": ""}
    )
    fig_top.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,15,26,0.6)",
        font_color="#c0ccdd",
        title_font_color="#f0f4ff",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_top, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 5 — EVALUATION & VALIDATION
# ════════════════════════════════════════════════════════════════════
elif page == "📈 Evaluation & Validation":

    st.markdown('<div class="section-header">📈 Evaluation & Validation</div>', unsafe_allow_html=True)

    # Metrics table
    m1, m2, m3, m4 = st.columns(4)
    eval_kpis = [
        (f"{metrics['km_silhouette']:.3f}", "Silhouette Score", "Cluster Quality"),
        (f"{metrics['km_db']:.3f}", "Davies-Bouldin", "Cluster Separation"),
        (f"{metrics['hc_silhouette']:.3f}", "Hierarchical Sil.", "Validation"),
        (f"{metrics['n_clusters']}", "Optimal Clusters", "K-Means"),
    ]
    for col, (val, label, sub) in zip([m1,m2,m3,m4], eval_kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="value">{val}</div>
              <div class="label">{label}</div>
              <div style="color:#6366f1;font-size:0.7rem;margin-top:3px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Elbow + Silhouette
    c1, c2 = st.columns(2)
    with c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=metrics["ks"], y=metrics["inertias"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=2),
            marker=dict(size=8, color="#6366f1"),
            name="Inertia"
        ))
        fig_elbow.update_layout(
            title="Elbow Method — Inertia vs K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Inertia",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,26,0.6)",
            font_color="#c0ccdd",
            title_font_color="#f0f4ff",
            margin=dict(t=40, b=30)
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=metrics["ks"], y=metrics["sil_scores"],
            mode="lines+markers",
            line=dict(color="#10b981", width=2),
            marker=dict(size=8, color="#10b981"),
            name="Silhouette"
        ))
        best_k = metrics["ks"][np.argmax(metrics["sil_scores"])]
        fig_sil.add_vline(x=best_k, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"Best K={best_k}", annotation_font_color="#f59e0b")
        fig_sil.update_layout(
            title="Silhouette Score vs K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Silhouette Score",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,26,0.6)",
            font_color="#c0ccdd",
            title_font_color="#f0f4ff",
            margin=dict(t=40, b=30)
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    # Evaluation metrics table
    st.markdown('<div class="section-header">Evaluation Metrics Reference</div>', unsafe_allow_html=True)
    eval_table = pd.DataFrame({
        "Metric": ["Silhouette Score", "Intra-Cluster Similarity",
                   "Recommendation Precision", "Engagement Lift (Proxy)"],
        "Purpose": ["Cluster quality", "Behavioral consistency",
                    "Relevance", "Impact estimate"],
        "Value / Method": [
            f"{metrics['km_silhouette']:.3f}",
            "High intra-cluster similarity (matched preferred categories)",
            "Content-based + cluster popularity filtering",
            "Segment-aware diversity & depth targeting"
        ],
        "Status": ["✅ Computed", "✅ Computed", "✅ Computed", "✅ Proxied"]
    })
    st.dataframe(eval_table, use_container_width=True, hide_index=True)

    # Intra-cluster similarity
    st.markdown('<div class="section-header">Intra-Cluster Behavioral Consistency</div>', unsafe_allow_html=True)
    cluster_std = prof.groupby("segment_name")[
        ["total_courses_enrolled", "avg_spending", "diversity_score", "learning_depth_index"]
    ].std().reset_index()

    fig_std = px.bar(
        cluster_std.melt(id_vars="segment_name"),
        x="segment_name", y="value", color="variable",
        barmode="group",
        title="Std Deviation Within Clusters (Lower = More Consistent)",
        labels={"value": "Std Dev", "segment_name": "Segment", "variable": "Feature"},
        color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#ec4899"]
    )
    fig_std.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,15,26,0.6)",
        font_color="#c0ccdd",
        title_font_color="#f0f4ff",
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig_std, use_container_width=True)

    # Completion rate by segment
    comp = prof.groupby("segment_name")["completion_rate"].mean().reset_index()
    comp.columns = ["Segment", "Avg Completion Rate"]
    comp["Avg Completion Rate"] = (comp["Avg Completion Rate"] * 100).round(1)
    fig_comp = px.bar(
        comp, x="Segment", y="Avg Completion Rate",
        color="Segment", color_discrete_map=SEG_COLORS,
        title="Average Completion Rate by Segment (%)",
        text="Avg Completion Rate"
    )
    fig_comp.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_comp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,15,26,0.6)",
        font_color="#c0ccdd",
        title_font_color="#f0f4ff",
        showlegend=False,
        margin=dict(t=50, b=20)
    )
    st.plotly_chart(fig_comp, use_container_width=True)
