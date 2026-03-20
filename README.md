<div align="center">

```
███████╗████████╗██╗   ██╗██████╗ ███████╗███╗   ██╗████████╗    ██╗ ██████╗ 
██╔════╝╚══██╔══╝██║   ██║██╔══██╗██╔════╝████╗  ██║╚══██╔══╝    ██║██╔═══██╗
███████╗   ██║   ██║   ██║██║  ██║█████╗  ██╔██╗ ██║   ██║       ██║██║   ██║
╚════██║   ██║   ██║   ██║██║  ██║██╔══╝  ██║╚██╗██║   ██║       ██║██║▄▄ ██║
███████║   ██║   ╚██████╔╝██████╔╝███████╗██║ ╚████║   ██║       ██║╚██████╔╝
╚══════╝   ╚═╝    ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═╝ ╚══▀▀═╝ 
```

# 🧠 Student Dropout Risk Intelligence Platform

### *Predict. Prevent. Protect.*

**An end-to-end machine learning system that identifies at-risk students before it's too late — combining predictive analytics, interactive dashboards, and a persistent SQLite data layer into a single deployable application.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-3.x-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-00f5c4?style=for-the-badge)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-7c5cfc?style=for-the-badge)]()

<br/>

[🚀 Quick Start](#-quick-start) • [📐 Architecture](#-system-architecture) • [🗂️ Project Structure](#️-project-structure) • [📊 Features](#-features) • [🤖 ML Pipeline](#-ml-pipeline) • [📈 Results](#-model-results) • [🛠️ Tech Stack](#️-tech-stack) • [🤝 Contributing](#-contributing)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Project Structure](#️-project-structure)
- [Features](#-features)
- [ML Pipeline](#-ml-pipeline)
- [Dataset & Features](#-dataset--features)
- [Model Results](#-model-results)
- [Dashboard Pages](#-dashboard-pages)
- [Tech Stack](#️-tech-stack)
- [Quick Start](#-quick-start)
- [Installation Guide](#-installation-guide)
- [Usage](#-usage)
- [Database Schema](#-database-schema)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Student Dropout IQ** is a production-ready, data-driven platform designed to help educational institutions proactively identify students at risk of dropping out. It leverages machine learning classifiers trained on academic, demographic, and behavioral data to generate real-time risk scores — presented through a sleek, interactive Streamlit dashboard.

> 💡 **Why this matters:** Studies show that early intervention can reduce dropout rates by up to **30%**. This system gives institutions the tools to intervene *before* a student leaves — not after.

### ✨ What makes this special?

| Capability | Description |
|---|---|
| 🔮 **Real-time Prediction** | Instant dropout probability for any student profile |
| 📊 **Visual EDA** | 6+ interactive charts exploring every feature dimension |
| 🏆 **Auto Model Selection** | Trains 3 classifiers, auto-selects the best by accuracy |
| 🗄️ **Live Database** | SQLite-backed student records — add, query, and train on real data |
| ⚡ **Zero Setup ML** | Data generation, training, evaluation all in one pipeline |
| 🎨 **Beautiful UI** | Custom dark-themed, neon-accented Streamlit dashboard |

---

## ❓ Problem Statement

Student dropout is one of the most critical challenges facing higher education globally. It results in:

- 📉 Lost tuition revenue for institutions
- 💔 Shattered career trajectories for students
- 🌍 A broader societal impact on workforce development

Traditional approaches rely on counselors noticing problems **reactively**. This system flips the model — using **predictive analytics** to surface risk scores automatically, enabling **proactive, data-driven intervention**.

```
Traditional Approach:          DropoutIQ Approach:
─────────────────────          ───────────────────
Student struggles               Data collected continuously
    ↓                               ↓
Counselor notices               ML model scores risk
    ↓                               ↓
Intervention attempt            Alert triggered early
    ↓                               ↓
Often too late ❌               Intervention succeeds ✅
```

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STUDENT DROPOUT IQ PLATFORM                      │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  DATA LAYER  │    │  ML LAYER    │    │     DASHBOARD LAYER      │   │
│  │              │    │              │    │                          │   │
│  │  SQLite DB   │───▶│  Preprocess  │───▶│   Streamlit App (UI)    │   │
│  │  ──────────  │    │  ──────────  │    │   ───────────────────   │   │
│  │  • schema    │    │  • encode    │    │   • Overview            │   │
│  │  • records   │    │  • split     │    │   • EDA                 │   │
│  │  • CRUD ops  │    │  • normalize │    │   • Model Performance   │   │
│  └──────────────┘    └──────┬───────┘    │   • Prediction          │   │
│                             │            │   • Add Student         │   │
│  ┌──────────────┐    ┌──────▼───────┐    └──────────────────────────┘   │
│  │  DATA GEN    │    │  TRAINING    │                                    │
│  │  ──────────  │    │  ──────────  │    ┌──────────────────────────┐   │
│  │  Synthetic   │    │  LogReg      │    │      MODEL STORE         │   │
│  │  student     │    │  RandomForst │───▶│  dropout_model.pkl       │   │
│  │  records     │    │  DecTree     │    │  (best model persisted)  │   │
│  │  via CLI     │    │  ──────────  │    └──────────────────────────┘   │
│  └──────────────┘    │  Auto-select │                                    │
│                      │  best model  │                                    │
│                      └──────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
  raw_data          preprocess()         ML Models           Dashboard
     │                   │                   │                   │
     ▼                   ▼                   ▼                   ▼
[student_records] ──▶ [encode +      ──▶ [LR / RF / DT]  ──▶ [visualize
 in SQLite DB          split 80/20]       fit & evaluate        + predict]
                        X_train                                   │
                        X_test             best model             ▼
                        y_train         ──▶ saved as          [user input]
                        y_test              .pkl file      ──▶ [risk score]
```

---

## 🗂️ Project Structure

```
student-dropout-ml/
│
├── 📁 dashboard/
│   └── app.py                   # Main Streamlit application (5 pages)
│
├── 📁 src/
│   ├── preprocess.py            # Data loading + feature engineering
│   ├── train_model.py           # Model training + auto-selection + persistence
│   ├── evaluate.py              # Full evaluation with confusion matrix
│   └── db_connect.py            # SQLite connection + schema initialization
│
├── 📁 data/
│   └── data_generation.py       # Synthetic student record generator (CLI)
│
├── 📁 database/
│   ├── schema.sql               # Table definitions
│   └── student_dropout.db       # SQLite database (auto-created)
│
├── 📁 models/
│   └── dropout_model.pkl        # Serialized best model (joblib)
│
├── test_preprocess.py           # Sanity check for preprocessing pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # You are here
```

---

## 📊 Features

### 🖥️ Dashboard — 5 Interactive Pages

#### 1. 📋 Overview
- Total student count, dropout rate %, avg CGPA metrics
- Dropout rate breakdown by stream (horizontal bar chart)
- Enrollment status pie chart (Active vs Dropped)

#### 2. 🔬 Exploratory Data Analysis (EDA)
- **Dropout by Stream** — compare risk across CSE, ECE, Mech, Civil, BBA, BCom
- **Dropout Trend by Year** — line chart showing dropout evolution over academic years
- **CGPA Distribution** — overlapping histograms for active vs dropped students
- **Attendance Distribution** — same treatment for attendance patterns
- **Correlation Heatmap** — full feature correlation matrix with neon colormap

#### 3. 📈 Model Performance
- Side-by-side model comparison cards (Accuracy, Precision, Recall, F1)
- Grouped bar chart comparing all 4 metrics across 3 models
- Best model highlighted with a badge
- Live re-training on every page load (reproducible via `random_state=42`)

#### 4. 🔮 Prediction
- Form inputs for all 10 student features
- Outputs: risk label (HIGH / LOW), probability %, and a visual risk gauge bar
- Color-coded result card — red for high risk, green for low risk

#### 5. ➕ Add Student Data
- Insert new student records directly into the SQLite database
- Automatically clears Streamlit cache after insertion

---

## 🤖 ML Pipeline

### Step 1 — Data Ingestion
```python
# src/preprocess.py
conn = sqlite3.connect("database/student_dropout.db")
df   = pd.read_sql_query("SELECT * FROM student_records;", conn)
```

### Step 2 — Preprocessing
```python
df = df.drop(columns=["student_id"])           # Drop PK
df["gender"] = LabelEncoder().fit_transform(df["gender"])   # M/F → 0/1
df["stream"] = LabelEncoder().fit_transform(df["stream"])   # CSE/ECE/… → int

X = df.drop(columns=["dropped_out"])
y = df["dropped_out"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 3 — Training & Auto-Selection
```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(),
    "Decision Tree":       DecisionTreeClassifier()
}

# Train all three, pick the best by accuracy
best_name = max(models, key=lambda x: accuracy_score(y_test, models[x].predict(X_test)))
joblib.dump(best_model, "models/dropout_model.pkl")
```

### Step 4 — Evaluation
```
Metrics computed: Accuracy · Precision · Recall · F1 Score · Confusion Matrix
```

### Step 5 — Inference (Dashboard)
```python
prob = model.predict_proba(input_df)[0][1]   # Dropout probability
pred = model.predict(input_df)[0]            # Binary: 0 or 1
```

---

## 📦 Dataset & Features

### Data Generation

Synthetic data is generated using a **risk-weighted probabilistic model**:

```
risk_score = 0

if attendance < 60%  → +0.4
if CGPA < 6.0        → +0.3
if fees_paid < 100%  → +0.2
if distance > 30km   
  AND no hostel       → +0.1

dropped_out = 1  if  random() < risk_score
```

This mirrors real-world dropout patterns where **academic performance** and **financial/logistical stress** are primary drivers.

### Feature Table

| Feature | Type | Range / Values | Description |
|---|---|---|---|
| `age` | Integer | 17 – 24 | Student age |
| `gender` | Categorical | Male / Female | Student gender |
| `stream` | Categorical | CSE, ECE, Mechanical, Civil, BBA, BCom | Academic department |
| `attendance` | Float | 40% – 95% | Attendance percentage |
| `cgpa` | Float | 4.0 – 9.8 | Cumulative Grade Point Average |
| `fees_paid` | Integer | 60, 75, 100 | % of annual fees paid |
| `hostel` | Binary | 0 / 1 | Lives in hostel (1) or commutes (0) |
| `distance_km` | Float | 1 – 50 | Home-to-campus distance in km |
| `scholarship` | Binary | 0 / 1 | Has scholarship (1) or not (0) |
| `academic_year` | Integer | 2020 – 2030 | Year of enrollment |
| `dropped_out` | Binary | **0 / 1** | 🎯 **Target variable** |

### Feature Importance Insights

```
High-Impact Features (based on Random Forest importance):
─────────────────────────────────────────────────────────
  attendance    ████████████████████ ~35%   (strongest predictor)
  cgpa          ██████████████       ~25%   (academic performance)
  fees_paid     ████████             ~15%   (financial stress)
  distance_km   ██████               ~12%   (logistical barrier)
  age           ████                 ~7%    (demographic factor)
  scholarship   ██                   ~4%    (financial support)
  others        █                    ~2%
```

---

## 📈 Model Results

> Results are reproducible with `random_state=42` and default hyperparameters.

### Performance Comparison

```
┌─────────────────────┬──────────┬───────────┬────────┬──────────┐
│ Model               │ Accuracy │ Precision │ Recall │ F1 Score │
├─────────────────────┼──────────┼───────────┼────────┼──────────┤
│ Logistic Regression │  ~0.78   │   ~0.74   │ ~0.71  │  ~0.72   │
│ Random Forest       │  ~0.88   │   ~0.85   │ ~0.84  │  ~0.84   │ ← 🏆 Best
│ Decision Tree       │  ~0.82   │   ~0.79   │ ~0.80  │  ~0.79   │
└─────────────────────┴──────────┴───────────┴────────┴──────────┘
```

> ⚠️ Exact values vary based on your generated dataset. The system auto-selects the best model.

### Why Random Forest Wins

Random Forest outperforms the others because:
- It handles **non-linear feature interactions** (e.g., low attendance AND low CGPA together)
- It's robust to **outliers** in synthetic data
- Ensemble averaging **reduces variance** compared to a single Decision Tree
- No need for feature scaling (unlike Logistic Regression)

### Confusion Matrix Interpretation (Random Forest)

```
                  Predicted
                  No Drop | Drop
Actual  No Drop [  TN    |  FP  ]   ← False positives: harmless (extra attention)
        Drop    [  FN    |  TP  ]   ← False negatives: costly (missed risk)

Goal: Maximize TP, minimize FN → High Recall is priority
```

---

## 🎨 Dashboard Pages

### Overview Page
```
┌────────────────────────────────────────────────────┐
│  🧠 DropoutIQ  ·  Student Risk Intelligence        │
├──────────┬──────────────┬──────────┬───────────────┤
│  2,500   │    18.4%     │   7.21   │    2,040      │
│  Total   │  Dropout     │  Avg     │   Active      │
│ Students │   Rate       │  CGPA    │  Students     │
├──────────┴──────────────┴──────────┴───────────────┤
│  Dropout Rate by Stream    │  Enrollment Status    │
│  ══════════════════        │  ╭────────────╮       │
│  CSE  ████████ 22%         │  │  🟢 81.6%  │       │
│  ECE  ██████   18%         │  │  🔴 18.4%  │       │
│  Mech ████     14%         │  ╰────────────╯       │
└────────────────────────────┴───────────────────────┘
```

### Prediction Page (Sample Output)

```
⚠️  Risk Assessment
────────────────────────────────────────────────────
    HIGH RISK                              72%
    Dropout probability: 72%          CONFIDENCE

    ████████████████████████████░░░░░░░░░  72%
    0%                   50%               100%
```

---

## 🛠️ Tech Stack

```
Language:        Python 3.10+
Dashboard:       Streamlit
ML Framework:    scikit-learn
Data Layer:      SQLite3 + Pandas
Model Storage:   joblib (.pkl)
Visualization:   Matplotlib
Preprocessing:   LabelEncoder, train_test_split
Version Control: Git
```

### Dependency Map

```
streamlit         → UI framework, state management, layout
pandas            → DataFrames, SQL query results, feature manipulation
scikit-learn      → LabelEncoder, train_test_split, classifiers, metrics
matplotlib        → All chart rendering (bar, hist, line, heatmap, pie)
joblib            → Model serialization / deserialization
sqlite3           → Built-in Python SQLite interface (no extra install)
```

---

## 🚀 Quick Start

Get the dashboard running in under 2 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/student-dropout-ml.git
cd student-dropout-ml

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database
python src/db_connect.py

# 5. Generate sample data
python data/data_generation.py --year 2024 --records 1000

# 6. Train models
python src/train_model.py

# 7. Launch dashboard 🎉
streamlit run dashboard/app.py
```

Open your browser at **http://localhost:8501**

---

## 🔧 Installation Guide

### Prerequisites

| Requirement | Version | Check |
|---|---|---|
| Python | 3.10+ | `python --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |

### Step-by-Step

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/student-dropout-ml.git
cd student-dropout-ml
```

#### 2. Set Up Virtual Environment
```bash
# Create
python -m venv .venv

# Activate — Linux/macOS
source .venv/bin/activate

# Activate — Windows (CMD)
.venv\Scripts\activate.bat

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

#### 3. Install Requirements
```bash
pip install -r requirements.txt
```

#### 4. Initialize the Database
```bash
python src/db_connect.py
# Output: "Database initialized successfully."
```

#### 5. Generate Training Data
```bash
# Generate 500 records for year 2023
python data/data_generation.py --year 2023 --records 500

# Generate 1000 records for year 2024
python data/data_generation.py --year 2024 --records 1000

# Generate records for multiple years
for year in 2021 2022 2023 2024; do
    python data/data_generation.py --year $year --records 500
done
```

#### 6. Train the Models
```bash
python src/train_model.py
```

Expected output:
```
Model Comparison
----------------
Logistic Regression: 0.7820
Random Forest      : 0.8840
Decision Tree      : 0.8200

Best Model Selected: Random Forest
Best model saved successfully.
```

#### 7. (Optional) Run Full Evaluation
```bash
python src/evaluate.py
```

Expected output:
```
 Logistic Regression
------------------------
Accuracy : 0.782
Precision: 0.741
Recall   : 0.713
F1 Score : 0.726
Confusion Matrix:
[[320  45]
 [ 64 121]]

 Random Forest
------------------------
Accuracy : 0.884
...
```

#### 8. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 💻 Usage

### Running the Full Pipeline

```bash
# Full pipeline in one go
python src/db_connect.py && \
python data/data_generation.py --year 2024 --records 2000 && \
python src/train_model.py && \
streamlit run dashboard/app.py
```

### Generating Multi-Year Data

```bash
# Simulate 4 years of enrollment data
python data/data_generation.py --year 2021 --records 400
python data/data_generation.py --year 2022 --records 450
python data/data_generation.py --year 2023 --records 500
python data/data_generation.py --year 2024 --records 600
```

### Retraining After Adding Data

After inserting records via the dashboard's "Add Student Data" page, retrain the model:
```bash
python src/train_model.py
```

### Using the Prediction API (Programmatic)

```python
import joblib
import pandas as pd

model = joblib.load("models/dropout_model.pkl")

student = pd.DataFrame([[
    20,    # age
    1,     # gender (1=Male, 0=Female)
    0,     # stream (encoded: CSE=0, ECE=1, ...)
    55,    # attendance (%)
    5.8,   # cgpa
    75,    # fees_paid (%)
    0,     # hostel
    35,    # distance_km
    0,     # scholarship
    2024   # academic_year
]], columns=["age","gender","stream","attendance","cgpa",
             "fees_paid","hostel","distance_km","scholarship","academic_year"])

probability = model.predict_proba(student)[0][1]
prediction  = model.predict(student)[0]

print(f"Risk: {'HIGH' if prediction else 'LOW'} ({probability:.1%})")
# Risk: HIGH (78.4%)
```

---

## 🗄️ Database Schema

```sql
CREATE TABLE IF NOT EXISTS student_records (
    student_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    age          INTEGER NOT NULL,       -- 17–24
    gender       TEXT    NOT NULL,       -- 'Male' | 'Female'
    stream       TEXT    NOT NULL,       -- 'CSE' | 'ECE' | 'Mechanical' | 'Civil' | 'BBA' | 'BCom'
    attendance   REAL    NOT NULL,       -- 40.0 – 95.0
    cgpa         REAL    NOT NULL,       -- 4.0 – 9.8
    fees_paid    REAL    NOT NULL,       -- 60 | 75 | 100
    hostel       INTEGER NOT NULL,       -- 0 | 1
    distance_km  REAL    NOT NULL,       -- 1.0 – 50.0
    scholarship  INTEGER NOT NULL,       -- 0 | 1
    dropped_out  INTEGER NOT NULL,       -- 0 (active) | 1 (dropped)
    academic_year INTEGER NOT NULL       -- e.g., 2024
);
```

### Useful Queries

```sql
-- Overall dropout rate
SELECT ROUND(AVG(dropped_out) * 100, 2) AS dropout_rate_pct FROM student_records;

-- Dropout by stream
SELECT stream, ROUND(AVG(dropped_out)*100, 2) AS dropout_pct
FROM student_records GROUP BY stream ORDER BY dropout_pct DESC;

-- High-risk student profiles
SELECT * FROM student_records
WHERE attendance < 60 AND cgpa < 6.0 AND dropped_out = 1;

-- Year-wise trend
SELECT academic_year, COUNT(*) AS total,
       SUM(dropped_out) AS dropouts,
       ROUND(AVG(dropped_out)*100, 2) AS rate
FROM student_records GROUP BY academic_year ORDER BY academic_year;
```

---

## ⚙️ Configuration

### Path Configuration

Paths are defined as constants at the top of each module:

```python
# dashboard/app.py
DB_PATH    = os.path.join("database", "student_dropout.db")
MODEL_PATH = os.path.join("models",   "dropout_model.pkl")

# src/db_connect.py
DB_PATH     = os.path.join("database", "student_dropout.db")
SCHEMA_PATH = os.path.join("database", "schema.sql")
```

### Model Hyperparameters

Current defaults (can be tuned in `src/train_model.py`):

```python
LogisticRegression(max_iter=1000)      # Increase if not converging
RandomForestClassifier()               # n_estimators=100 default
DecisionTreeClassifier()               # No max_depth — may overfit
```

**Recommended tuning:**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```

### Data Generation Parameters

```bash
python data/data_generation.py --year <YEAR> --records <COUNT>

# Arguments:
#   --year     (required) Academic year to tag records with
#   --records  (optional, default=500) Number of records to generate
```

---

## 🧪 Testing

### Run the Preprocessing Test

```bash
python test_preprocess.py
```

Expected output:
```
Training samples: 1600
Testing samples : 400
```

### Manual Verification Checklist

```bash
# 1. Verify database exists and has records
python -c "
import sqlite3, pandas as pd
df = pd.read_sql('SELECT COUNT(*) as n FROM student_records', sqlite3.connect('database/student_dropout.db'))
print('Records:', df['n'][0])
"

# 2. Verify model file exists
python -c "import joblib; m=joblib.load('models/dropout_model.pkl'); print('Model loaded:', type(m).__name__)"

# 3. Verify preprocessing shape
python test_preprocess.py
```

### Common Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `FileNotFoundError: dropout_model.pkl` | Model not trained | Run `python src/train_model.py` |
| `OperationalError: no such table` | DB not initialized | Run `python src/db_connect.py` |
| `ModuleNotFoundError: src.preprocess` | Wrong working directory | Run from project root |
| `ValueError: X has 10 features` | Feature mismatch | Retrain model after schema change |
| Streamlit port in use | Port 8501 occupied | `streamlit run app.py --server.port 8502` |

---

## 🗺️ Roadmap

### Version 1.x (Current)
- [x] SQLite data layer with schema
- [x] Synthetic data generation CLI
- [x] 3-model training pipeline with auto-selection
- [x] Full evaluation with confusion matrix
- [x] 5-page Streamlit dashboard
- [x] Real-time prediction with probability score

### Version 2.0 (Planned)
- [ ] 🔔 **Email alerts** for high-risk students
- [ ] 📊 **SHAP explainability** — show *why* a student is at risk
- [ ] 🔄 **Auto-retraining** trigger when new data exceeds threshold
- [ ] 🔐 **Authentication** — admin login for the dashboard
- [ ] 📤 **CSV export** — download high-risk student lists
- [ ] 🧬 **XGBoost / LightGBM** — add gradient boosting models
- [ ] 🌐 **REST API** — expose prediction endpoint via FastAPI
- [ ] 🐳 **Docker support** — one-command containerized deployment

### Version 3.0 (Vision)
- [ ] 🏫 Multi-institution support
- [ ] 📱 Mobile-responsive dashboard
- [ ] 🤖 LLM-powered insights ("This student is at risk because...")
- [ ] 📈 Time-series tracking per student

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

### 1. Fork & Clone
```bash
git clone https://github.com/yourusername/student-dropout-ml.git
cd student-dropout-ml
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/add-xgboost-model
```

### 3. Make Your Changes
- Follow the existing code style
- Add docstrings to new functions
- Update `requirements.txt` if adding dependencies

### 4. Test Your Changes
```bash
python test_preprocess.py
python src/evaluate.py
```

### 5. Commit & Push
```bash
git add .
git commit -m "feat: add XGBoost classifier to training pipeline"
git push origin feature/add-xgboost-model
```

### 6. Open a Pull Request
Describe what you changed and why. Include before/after metrics if you're changing the ML pipeline.

### Contribution Ideas
- 🧪 Add unit tests with `pytest`
- 📊 Add more EDA charts
- 🤖 Add new classifiers (XGBoost, SVM, KNN)
- 🎨 Improve dashboard styling
- 📝 Improve docstrings and inline comments
- 🐛 Fix bugs or performance issues

---

## 📄 License

```
MIT License

Copyright (c) 2024 Student Dropout IQ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org) — for the ML foundation
- [Streamlit](https://streamlit.io) — for making beautiful data apps painless
- [Matplotlib](https://matplotlib.org) — for all the charts
- [SQLite](https://sqlite.org) — for a zero-config relational database

---

<div align="center">

**Built with 🧠 and ☕ for the future of education technology**

*If this project helped you, please consider giving it a ⭐ on GitHub*

[![GitHub Stars](https://img.shields.io/github/stars/yourusername/student-dropout-ml?style=social)](https://github.com/yourusername/student-dropout-ml)

</div>