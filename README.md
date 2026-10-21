# EduPredict-XAI 🎓

## Explainable AI for Student Performance Prediction

EduPredict-XAI is a research-oriented Explainable Artificial Intelligence (XAI) system designed to predict student final exam performance and explain the factors influencing each prediction.

The system combines machine learning, TabPFN regression, SHAP-based explainability, and personalized intervention recommendations into an interactive Streamlit application.

---

## 🎯 Project Objective

The main objective of EduPredict-XAI is not only to predict a student's future academic performance but also to answer:

> "Why did the model make this prediction?"

The system therefore provides:

1. Predicted final exam score
2. Performance category
3. Important factors influencing the prediction
4. SHAP-based local explanations
5. Personalized intervention recommendations

---

## 🔬 Research Focus

The project focuses on Explainable AI for educational prediction.

Instead of treating the machine learning model as a black box, EduPredict-XAI provides interpretable explanations for individual student predictions.

The research workflow is:

Student Data
↓
Feature Engineering
↓
Train/Test Split
↓
Baseline Models
↓
TabPFN Regression
↓
Model Evaluation
↓
SHAP Explanation
↓
Performance Categorization
↓
Personalized Recommendations

---

## 🧠 Machine Learning Model

### Primary Model

**TabPFN Regressor**

TabPFN is used as the primary model for predicting the student's:

**Final Exam Score**

Target variable:

`final_exam_score`

---

## 📊 Input Features

The current system uses the following student features:

- Gender
- Study Time Hours
- Attendance Percentage
- Sleep Hours
- Parental Education
- Internet Access
- Extracurricular Activities
- Part-time Job
- Previous Grade

---

## 🤖 Baseline Models

The TabPFN model is compared against traditional gradient-boosting models:

- CatBoost
- XGBoost
- LightGBM

Model performance is evaluated using:

- MAE
- RMSE
- R²

The current evaluation identified **TabPFN as the best-performing model based on RMSE**.

---

## 🔍 Explainable AI

### SHAP

SHAP (SHapley Additive exPlanations) is used to explain the model predictions.

The project provides both:

### Global Explanation

Identifies which student features are generally important to the model.

### Local Explanation

Explains why the model predicted a particular score for an individual student.

For example:

Predicted Score: 94.82

Important positive factors may include:

- High attendance
- Study time
- Previous grade
- Internet access

The SHAP explanation therefore provides a student-specific interpretation of the prediction.

---

## 💡 Personalized Intervention Recommendations

The project extends XAI beyond explanation.

The SHAP factors are combined with the student's performance category to generate personalized intervention recommendations.

Example:

Low attendance
↓
SHAP identifies attendance as an important negative factor
↓
System generates recommendation
↓
"Improve attendance and maintain regular class participation."

This creates a complete pipeline:

**Prediction → Explanation → Intervention**

---

## 🖥️ Application

The project includes an interactive Streamlit application.

The user enters student information such as:

- Study time
- Attendance
- Sleep
- Previous grade
- Internet access
- Parental education
- Extracurricular activities
- Part-time job

The application then produces:

### Prediction

Predicted final exam score.

### Performance Category

For example:

- Excellent
- Good
- Average
- Needs Improvement
- At Risk

### SHAP Factors

Features that contributed positively or negatively to the prediction.

### Personalized Recommendations

Actionable suggestions based on the student's characteristics and model explanation.

---

## 🏗️ Project Structure

```text
EduPredict-XAI/
│
├── app/
│   └── App.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── catboost_baseline.cbm
│   ├── xgboost_baseline.pkl
│   ├── lightgbm_baseline.pkl
│   └── tabpfn_fitted_model.pkl
│
├── results/
│   ├── xai/
│   └── model evaluation results
│
├── notebooks/
│   ├── 01_*.ipynb
│   ├── 02_*.ipynb
│   ├── ...
│   └── 10_*.ipynb
│
├── src/
│   ├── model training scripts
│   ├── SHAP explanation scripts
│   └── recommendation scripts
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Technologies Used
Python, Pandas, NumPy, Scikit-learn, TabPFN, CatBoost, XGBoost, LightGBM, SHAP, Streamlit, Matplotlib, SciPy

## 📈 Evaluation Metrics

1. Mean Absolute Error (MAE) - Measures the average absolute difference between actual and predicted scores.

2. Root Mean Squared Error (RMSE) - Measures prediction error while giving greater weight to larger errors.

3. R² Score - Measures how much of the variation in final exam scores is explained by the model.

## ▶️ Run the Project
```bash
# Windows: .venv\Scripts\activate     
pip install -r requirements.txt
streamlit run app/App.py
```

## 📜 License

This project is intended for academic and research purposes.

---

## 🚀 Full Upgrade (backend + frontend)

The core idea is unchanged: **Prediction → Explanation → Intervention** with TabPFN + SHAP.
Everything around it has been rebuilt.

### Frontend: 
| Page | What it offers |
|------|----------------|
| 📊 Dashboard | KPIs, score distribution, study-time effect, risk-band comparison, model leaderboard |
| 🔮 Predict & Explain | Presets, predicted score, **90% interval**, cohort percentile, **SHAP waterfall** + plain-language explanation, **ranked action plan with estimated point gains**, **live what-if simulator**, HTML/JSON report download |
| 📂 Class / Batch Analysis | CSV upload with validation, class-wide scoring, early-warning flags, filters, download |
| 🧠 Research Insights | One combined page with four tabs: **Model performance** (cross-validation, hold-out, residuals), **Explainability** (method agreement, stability, faithfulness, PDP/ICE), **Uncertainty & risk** (conformal coverage, risk bands, early warning), **Fairness & ethics** (subgroup audit, gender-flip test) |
| ℹ️ About | Project overview, input factors, methods, glossary, FAQ, limitations |

### Backend
- `src/service.py` – one `EduPredictService` used by the app, the API and the tests: validation, prediction, conformal intervals, cohort bands, SHAP (exact TreeSHAP or permutation SHAP for TabPFN), what-if, batch scoring.
- `src/recommendations.py` – new **data-driven** action engine. The old rules fired only when a raw threshold was crossed, so most students got a generic message. The new engine proposes realistic targets and asks the model how many points each change would add.
- `api/main.py` – **REST API** (FastAPI): `/health`, `/predict`, `/explain`, `/whatif`, `/batch` with input validation. Run `uvicorn api.main:app --reload`, docs at `/docs`.
- `src/model_loader.py` – safe loading with a CatBoost fallback so a fresh clone never crashes.

### Run
```bash
pip install -r requirements.txt
streamlit run app/App.py                 # web app
uvicorn api.main:app --reload            # optional REST API
python -m pytest tests                   # tests
python run_all_upgrades.py               # regenerate research results
```

### Key findings
- A simple linear model is statistically as accurate as TabPFN on this dataset (RMSE 6.32 vs 6.45, p=0.19): the signal is mostly linear. TabPFN beats XGBoost, LightGBM and Extra Trees significantly.
- 90% conformal intervals cover 91.1% of unseen students (about ±10 points).
- SHAP rankings agree across 5 methods (mean Spearman 0.95) and are stable under bootstrap.
- Fixed 75/60/50/35 thresholds label 78.8% of students "Excellent"; cohort-relative bands fix this.
