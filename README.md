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

## ⚙️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
TabPFN
CatBoost
XGBoost
LightGBM
SHAP
Streamlit
Matplotlib

## 📈 Evaluation Metrics

1. Mean Absolute Error (MAE) - Measures the average absolute difference between actual and predicted scores.

2. Root Mean Squared Error (RMSE) - Measures prediction error while giving greater weight to larger errors.

3. R² Score - Measures how much of the variation in final exam scores is explained by the model.

## ▶️ Run the Project
.venv\Scripts\activate
streamlit run app/App.py

## 📜 License

This project is intended for academic and research purposes.