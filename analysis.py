"""
EduPro — Student Segmentation & Personalized Course Recommendation
Core Analysis Module
===================
Steps:
  1. Load & merge datasets
  2. Feature engineering (engagement, preference, behavioral)
  3. Data preprocessing (normalize, encode)
  4. Learner segmentation via K-Means + Hierarchical validation
  5. Cluster interpretation & naming
  6. Personalized recommendation logic
  7. Evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════
def load_data(data_dir="data"):
    users        = pd.read_csv(f"{data_dir}/users.csv")
    courses      = pd.read_csv(f"{data_dir}/courses.csv")
    transactions = pd.read_csv(f"{data_dir}/transactions.csv")
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
    return users, courses, transactions


# ═══════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
def engineer_features(users, courses, transactions):
    # Merge transactions with course metadata
    txn = transactions.merge(courses, on="CourseID", how="left")

    # ── Engagement Features ──────────────────────────────────────
    eng = txn.groupby("UserID").agg(
        total_courses_enrolled=("CourseID", "count"),
        unique_categories=("CourseCategory", "nunique"),
        avg_courses_per_category=("CourseCategory",
                                   lambda x: x.count() / x.nunique()),
        enrollment_frequency=("TransactionDate",
                               lambda x: (x.max() - x.min()).days / max(x.count() - 1, 1)),
        completion_rate=("CompletionStatus",
                          lambda x: (x == "Completed").sum() / len(x)),
    ).reset_index()

    # ── Preference Features ───────────────────────────────────────
    pref_cat   = (txn.groupby(["UserID", "CourseCategory"])
                     .size().reset_index(name="cnt"))
    top_cat    = (pref_cat.sort_values("cnt", ascending=False)
                          .groupby("UserID").first()
                          .rename(columns={"CourseCategory": "preferred_category"})
                          .reset_index()[["UserID", "preferred_category"]])

    pref_level = (txn.groupby(["UserID", "CourseLevel"])
                     .size().reset_index(name="cnt"))
    top_level  = (pref_level.sort_values("cnt", ascending=False)
                             .groupby("UserID").first()
                             .rename(columns={"CourseLevel": "preferred_level"})
                             .reset_index()[["UserID", "preferred_level"]])

    avg_rating_enrolled = (txn.groupby("UserID")["CourseRating"]
                               .mean().reset_index()
                               .rename(columns={"CourseRating": "avg_course_rating_enrolled"}))

    # ── Behavioral Features ───────────────────────────────────────
    beh = txn.groupby("UserID").agg(
        avg_spending=("Amount", "mean"),
        total_spending=("Amount", "sum"),
        diversity_score=("CourseCategory", "nunique"),  # categories explored
        avg_duration_enrolled=("CourseDurationHrs", "mean"),
    ).reset_index()

    # Learning depth index: ratio of advanced to beginner courses
    depth = txn.copy()
    depth["is_advanced"] = (depth["CourseLevel"] == "Advanced").astype(int)
    depth["is_beginner"]  = (depth["CourseLevel"] == "Beginner").astype(int)
    depth_agg = depth.groupby("UserID").agg(
        adv_count=("is_advanced", "sum"),
        beg_count=("is_beginner", "sum")
    ).reset_index()
    depth_agg["learning_depth_index"] = (
        depth_agg["adv_count"] / (depth_agg["beg_count"] + 1)
    )

    # ── Merge all features ────────────────────────────────────────
    profiles = (users[["UserID", "Age", "Gender"]]
                .merge(eng,               on="UserID", how="left")
                .merge(top_cat,           on="UserID", how="left")
                .merge(top_level,         on="UserID", how="left")
                .merge(avg_rating_enrolled, on="UserID", how="left")
                .merge(beh,               on="UserID", how="left")
                .merge(depth_agg[["UserID", "learning_depth_index"]],
                       on="UserID", how="left"))

    # Fill numeric NaN with 0, string NaN with "Unknown"
    for col in profiles.columns:
        if profiles[col].dtype == object:
            profiles[col] = profiles[col].fillna("Unknown")
        else:
            profiles[col] = profiles[col].fillna(0)
    return profiles, txn


# ═══════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
def preprocess(profiles):
    df = profiles.copy()

    # Encode categoricals
    le_gender   = LabelEncoder()
    le_cat      = LabelEncoder()
    le_level    = LabelEncoder()

    df["gender_enc"]    = le_gender.fit_transform(df["Gender"].astype(str))
    df["pref_cat_enc"]  = le_cat.fit_transform(df["preferred_category"].astype(str))
    df["pref_lvl_enc"]  = le_level.fit_transform(df["preferred_level"].astype(str))

    numerical_cols = [
        "Age", "total_courses_enrolled", "avg_courses_per_category",
        "unique_categories", "completion_rate",
        "avg_course_rating_enrolled", "avg_spending", "total_spending",
        "diversity_score", "learning_depth_index", "avg_duration_enrolled",
        "gender_enc", "pref_cat_enc", "pref_lvl_enc"
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[numerical_cols])

    return X, numerical_cols, scaler, df


# ═══════════════════════════════════════════════════════════════════
# 4. CLUSTER SELECTION — ELBOW + SILHOUETTE
# ═══════════════════════════════════════════════════════════════════
def find_optimal_k(X, k_range=range(2, 11)):
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
    return list(k_range), inertias, sil_scores


# ═══════════════════════════════════════════════════════════════════
# 5. CLUSTERING
# ═══════════════════════════════════════════════════════════════════
def cluster_learners(X, n_clusters=4):
    # Primary: K-Means
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)
    km_sil    = silhouette_score(X, km_labels)
    km_db     = davies_bouldin_score(X, km_labels)

    # Validation: Hierarchical
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    hc_labels = hc.fit_predict(X)
    hc_sil    = silhouette_score(X, hc_labels)

    return km_labels, km, km_sil, km_db, hc_labels, hc_sil


# ═══════════════════════════════════════════════════════════════════
# 6. CLUSTER NAMING
# ═══════════════════════════════════════════════════════════════════
SEGMENT_NAMES = {
    0: "🧭 Explorers",       # high diversity, many categories
    1: "🎯 Specialists",     # high depth, few categories
    2: "🏆 Career Climbers", # high spending, certification focus
    3: "🌱 Casual Learners"  # low engagement overall
}

SEGMENT_DESC = {
    0: "Broad explorers sampling courses across many domains at beginner level.",
    1: "Deep specialists who advance steadily in a single subject area.",
    2: "Career-driven learners investing in certifications and advanced content.",
    3: "Occasional learners with low engagement and limited course history."
}


def name_clusters(profiles_df, km_labels):
    """
    Assign human-readable names based on cluster centroid characteristics.
    We rank clusters by diversity_score to map them to segment names.
    """
    df = profiles_df.copy()
    df["cluster"] = km_labels

    cluster_stats = df.groupby("cluster").agg(
        diversity_score=("diversity_score", "mean"),
        learning_depth_index=("learning_depth_index", "mean"),
        avg_spending=("avg_spending", "mean"),
        total_courses_enrolled=("total_courses_enrolled", "mean"),
        completion_rate=("completion_rate", "mean"),
    )

    # Sort by diversity desc -> Explorers get highest diversity
    sorted_by_diversity = cluster_stats.sort_values("diversity_score", ascending=False).index.tolist()
    sorted_by_depth     = cluster_stats.sort_values("learning_depth_index", ascending=False).index.tolist()
    sorted_by_spending  = cluster_stats.sort_values("avg_spending", ascending=False).index.tolist()

    cluster_name_map = {}
    # Explorers = highest diversity
    cluster_name_map[sorted_by_diversity[0]] = 0
    # Specialists = highest depth (among remaining)
    for c in sorted_by_depth:
        if c not in cluster_name_map:
            cluster_name_map[c] = 1
            break
    # Career Climbers = highest spending (among remaining)
    for c in sorted_by_spending:
        if c not in cluster_name_map:
            cluster_name_map[c] = 2
            break
    # Casual = rest
    for c in cluster_stats.index:
        if c not in cluster_name_map:
            cluster_name_map[c] = 3

    df["segment_id"]   = df["cluster"].map(cluster_name_map)
    df["segment_name"] = df["segment_id"].map(SEGMENT_NAMES)
    df["segment_desc"] = df["segment_id"].map(SEGMENT_DESC)
    return df, cluster_name_map


# ═══════════════════════════════════════════════════════════════════
# 7. RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════
def build_recommendations(profiles_df, courses, txn, top_n=5):
    """
    Cluster-aware recommendations:
      - Content-based: match preferred category & level
      - Popularity within cluster: weighted by rating
      - Exclude already-enrolled courses
    Returns dict {UserID: [CourseID, ...]}
    """
    enrolled_map = txn.groupby("UserID")["CourseID"].apply(set).to_dict()
    recommendations = {}

    for _, user in profiles_df.iterrows():
        uid          = user["UserID"]
        pref_cat     = user.get("preferred_category", None)
        pref_level   = user.get("preferred_level", None)
        segment_id   = user.get("segment_id", 3)
        enrolled     = enrolled_map.get(uid, set())

        # Segment-specific strategy
        if segment_id == 0:   # Explorers — diverse categories, beginner+intermediate
            candidates = courses[courses["CourseLevel"].isin(["Beginner", "Intermediate"])]
        elif segment_id == 1: # Specialists — same category, advancing level
            candidates = courses[courses["CourseCategory"] == pref_cat]
        elif segment_id == 2: # Career Climbers — advanced, project-based
            candidates = courses[
                (courses["CourseLevel"] == "Advanced") |
                (courses["CourseType"] == "Project-Based")
            ]
        else:                 # Casual — high rated, short courses
            candidates = courses[courses["CourseDurationHrs"] <= 10]

        # Exclude already enrolled
        candidates = candidates[~candidates["CourseID"].isin(enrolled)]

        # Score = rating (with diversity bonus for explorers)
        candidates = candidates.copy()
        candidates["score"] = candidates["CourseRating"]
        if segment_id == 0:
            # Bonus for unexplored categories
            explored_cats = txn[txn["UserID"] == uid]["CourseCategory"].unique()
            candidates["score"] += candidates["CourseCategory"].apply(
                lambda c: 0.3 if c not in explored_cats else 0
            )

        top = candidates.nlargest(top_n, "score")[
            ["CourseID", "CourseName", "CourseCategory", "CourseLevel",
             "CourseRating", "CourseDurationHrs", "CoursePrice"]
        ]
        recommendations[uid] = top.to_dict("records")

    return recommendations


# ═══════════════════════════════════════════════════════════════════
# 8. PCA FOR VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
def compute_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


# ═══════════════════════════════════════════════════════════════════
# MAIN — run full pipeline
# ═══════════════════════════════════════════════════════════════════
def run_pipeline(data_dir="data", n_clusters=4):
    print("📦 Loading data...")
    users, courses, transactions = load_data(data_dir)

    print("🔧 Engineering features...")
    profiles, txn = engineer_features(users, courses, transactions)

    print("⚙️  Preprocessing...")
    X, feature_cols, scaler, profiles_enc = preprocess(profiles)

    print("🔍 Finding optimal K...")
    ks, inertias, sil_scores = find_optimal_k(X)
    best_k = ks[np.argmax(sil_scores)]
    print(f"   Best K by silhouette: {best_k} (score={max(sil_scores):.3f})")

    print(f"🎯 Clustering with K={n_clusters}...")
    km_labels, km_model, km_sil, km_db, hc_labels, hc_sil = cluster_learners(X, n_clusters)
    print(f"   K-Means  Silhouette={km_sil:.3f}  DB={km_db:.3f}")
    print(f"   Hierarchical Silhouette={hc_sil:.3f}")

    print("🏷️  Naming clusters...")
    profiles_enc, cluster_name_map = name_clusters(profiles_enc, km_labels)

    print("🤝 Building recommendations...")
    recs = build_recommendations(profiles_enc, courses, txn)

    print("📐 PCA for visualization...")
    X_pca, pca = compute_pca(X)
    profiles_enc["pca_x"] = X_pca[:, 0]
    profiles_enc["pca_y"] = X_pca[:, 1]

    print("✅ Pipeline complete!")
    return {
        "users": users,
        "courses": courses,
        "transactions": txn,
        "profiles": profiles_enc,
        "recommendations": recs,
        "km_model": km_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": {
            "km_silhouette": km_sil,
            "km_db": km_db,
            "hc_silhouette": hc_sil,
            "ks": ks,
            "inertias": inertias,
            "sil_scores": sil_scores,
            "n_clusters": n_clusters,
        }
    }


if __name__ == "__main__":
    result = run_pipeline()
    prof = result["profiles"]
    print("\n── Segment Distribution ──")
    print(prof["segment_name"].value_counts())
    print("\n── Sample Profile ──")
    print(prof[["UserID", "segment_name", "total_courses_enrolled",
                "diversity_score", "avg_spending"]].head(5))
