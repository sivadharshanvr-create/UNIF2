"""
EduPro Dataset Generator
Generates synthetic users, courses, and transaction data for the
Student Segmentation & Personalized Course Recommendation project.
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N_USERS = 1000
N_COURSES = 150
N_TRANSACTIONS = 5000

# ── Users ──────────────────────────────────────────────────────────────────
genders = ["Male", "Female", "Non-binary", "Prefer not to say"]
users = pd.DataFrame({
    "UserID": [f"U{str(i).zfill(4)}" for i in range(1, N_USERS + 1)],
    "Age": np.random.randint(18, 65, N_USERS),
    "Gender": np.random.choice(genders, N_USERS, p=[0.48, 0.44, 0.05, 0.03]),
    "Country": np.random.choice(
        ["India", "USA", "UK", "Canada", "Germany", "Australia", "Brazil"],
        N_USERS, p=[0.35, 0.20, 0.10, 0.10, 0.10, 0.08, 0.07]
    ),
    "JoinDate": [
        (datetime(2022, 1, 1) + timedelta(days=random.randint(0, 900))).strftime("%Y-%m-%d")
        for _ in range(N_USERS)
    ]
})

# ── Courses ─────────────────────────────────────────────────────────────────
categories = ["Data Science", "Web Development", "Business", "Design",
              "Marketing", "AI/ML", "Cloud", "Cybersecurity", "Finance", "Language"]
types_ = ["Video", "Interactive", "Project-Based", "Live"]
levels = ["Beginner", "Intermediate", "Advanced"]

courses = pd.DataFrame({
    "CourseID": [f"C{str(i).zfill(3)}" for i in range(1, N_COURSES + 1)],
    "CourseName": [f"Course_{i}" for i in range(1, N_COURSES + 1)],
    "CourseCategory": np.random.choice(categories, N_COURSES),
    "CourseType": np.random.choice(types_, N_COURSES, p=[0.4, 0.25, 0.25, 0.10]),
    "CourseLevel": np.random.choice(levels, N_COURSES, p=[0.40, 0.35, 0.25]),
    "CourseRating": np.round(np.random.uniform(3.0, 5.0, N_COURSES), 1),
    "CourseDurationHrs": np.random.randint(2, 60, N_COURSES),
    "CoursePrice": np.round(np.random.uniform(9.99, 199.99, N_COURSES), 2),
})

# ── Transactions ─────────────────────────────────────────────────────────────
user_ids  = users["UserID"].tolist()
course_ids = courses["CourseID"].tolist()

transactions = pd.DataFrame({
    "TransactionID": [f"T{str(i).zfill(5)}" for i in range(1, N_TRANSACTIONS + 1)],
    "UserID":   np.random.choice(user_ids,   N_TRANSACTIONS),
    "CourseID": np.random.choice(course_ids, N_TRANSACTIONS),
    "TransactionDate": [
        (datetime(2022, 1, 1) + timedelta(days=random.randint(0, 900))).strftime("%Y-%m-%d")
        for _ in range(N_TRANSACTIONS)
    ],
    "Amount":   np.round(np.random.uniform(9.99, 199.99, N_TRANSACTIONS), 2),
    "CompletionStatus": np.random.choice(
        ["Completed", "In Progress", "Not Started"],
        N_TRANSACTIONS, p=[0.45, 0.35, 0.20]
    ),
    "Rating": np.where(
        np.random.rand(N_TRANSACTIONS) > 0.3,
        np.round(np.random.uniform(2.5, 5.0, N_TRANSACTIONS), 1),
        np.nan
    )
})

# Remove duplicate user-course pairs (keep first)
transactions = transactions.drop_duplicates(subset=["UserID", "CourseID"])

# Save
users.to_csv("/home/claude/edupro/data/users.csv", index=False)
courses.to_csv("/home/claude/edupro/data/courses.csv", index=False)
transactions.to_csv("/home/claude/edupro/data/transactions.csv", index=False)

print(f"✅ Users: {len(users)}")
print(f"✅ Courses: {len(courses)}")
print(f"✅ Transactions: {len(transactions)}")
print("Data saved to /home/claude/edupro/data/")
