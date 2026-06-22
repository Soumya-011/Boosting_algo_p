# 🎓 Student Exam Score Predictor

## 📌 Overview
The **Student Exam Score Predictor** is an end-to-end Machine Learning project designed to forecast a student's final exam performance based on their demographics, study habits, and environmental factors. 

The project utilizes a highly optimized **XGBoost Regressor** to capture complex, non-linear relationships in the data. The final model is deployed as an interactive web application using **Streamlit**, allowing users to input student details and receive instant score predictions.

---

## 🚀 Features
* **Advanced Data Preprocessing:** Handles both Ordinal encoding (e.g., Sleep Quality) and One-Hot Encoding (e.g., Study Method) perfectly aligned for the model.
* **Smart Feature Selection:** Automatically ignores useless identifiers (like `student_id`) to prevent data leakage and overfitting.
* **Hyperparameter Tuning:** Uses `RandomizedSearchCV` to find the absolute best mathematical constraints for the XGBoost algorithm.
* **Interactive Web App:** A clean, user-friendly UI built with Streamlit.

---

## 📂 Project Structure

| File Name | Description |
| :--- | :--- |
| `Exam_Score.csv` | The raw dataset containing student records. |
| `Boosting_Algo_practice.ipynb` | The Jupyter Notebook containing all EDA, preprocessing, model training, and hyperparameter tuning. |
| `app.py` | The Streamlit web application script. |
| `requirements.txt` | The list of Python dependencies required to run the app. |
| `best_xgb_model.pkl` | The exported, fully-trained XGBoost model. *(Generated after running the notebook)* |
| `expected_columns.pkl` | A saved list of the exact column structure required by the model. *(Generated after running the notebook)* |

---

## 🛠️ Installation & Setup

### 1. Clone or Download the Repository
Ensure all the files listed in the project structure are in the same folder on your local machine.

### 2. Install Dependencies
Open your terminal or command prompt, navigate to the project folder, and run the following command to install all required libraries:
```bash
pip install -r requirements.txt
