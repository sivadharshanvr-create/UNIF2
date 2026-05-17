# 🎓 EduPro — Student Segmentation & Personalized Course Recommendation System

A complete Data Science project for the EduPro online learning platform.

## Project Structure

```
edupro/
├── data/
│   ├── users.csv           # 1,000 learner profiles
│   ├── courses.csv         # 150 courses across 10 categories
│   └── transactions.csv    # 4,930 enrollment records
├── generate_data.py        # Synthetic dataset generator
├── analysis.py             # Core ML pipeline (feature eng + clustering + recs)
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
└── EduPro_Research_Paper.docx  # Full research paper
```

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python generate_data.py
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Run analysis only
```bash
python analysis.py
```

## Key Features

### Machine Learning Pipeline (`analysis.py`)
- **Feature Engineering**: 14 learner-level features (engagement, preference, behavioral)
- **Preprocessing**: StandardScaler normalization + LabelEncoding
- **Cluster Selection**: Elbow + Silhouette methods (K=2..10)
- **K-Means Clustering**: K=4, validated against Hierarchical Clustering
- **Recommendation Engine**: Cluster-aware + content-based + rating-weighted

### Learner Segments
| Segment | Description |
|---------|-------------|
| 🧭 Explorers | Broad samplers, high diversity, beginner/intermediate |
| 🎯 Specialists | Deep focus in one subject, advancing level |
| 🏆 Career Climbers | High spenders, advanced + project-based content |
| 🌱 Casual Learners | Low engagement, prefer short high-rated courses |

### Evaluation Metrics
| Metric | Value |
|--------|-------|
| Silhouette Score (K-Means) | 0.093 |
| Davies-Bouldin Score | 2.427 |
| Silhouette Score (Hierarchical) | 0.111 |

### Streamlit Dashboard Pages
1. **Overview Dashboard** — KPIs, PCA cluster scatter, distributions
2. **Cluster Explorer** — Segment cards, radar chart, heatmap
3. **Learner Profile** — Individual deep-dive with enrollment history
4. **Course Recommendations** — Personalized with level/category filters
5. **Evaluation & Validation** — Elbow, silhouette curves, metrics table

## Dataset Fields

### Users Sheet
- UserID, Age, Gender, Country, JoinDate

### Courses Sheet
- CourseID, CourseName, CourseCategory, CourseType, CourseLevel, CourseRating, CourseDurationHrs, CoursePrice

### Transactions Sheet
- TransactionID, UserID, CourseID, TransactionDate, Amount, CompletionStatus, Rating

## Submission Checklist
- [x] Research paper (EDA, insights, recommendations) → `EduPro_Research_Paper.docx`
- [x] Streamlit dashboard (live analytics) → `streamlit run app.py`
- [x] GitHub repository → push all files
- [x] Executive summary → included in research paper Section 9
