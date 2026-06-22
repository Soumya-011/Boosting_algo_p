import streamlit as st
import pandas as pd
import pickle

# --- Load the Saved Pickles ---
@st.cache_resource
def load_components():
    with open('best_xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('expected_columns.pkl', 'rb') as file:
        expected_columns = pickle.load(file)
    return model, expected_columns

model, expected_columns = load_components()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Student Exam Score Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Exam Score Predictor")
st.write("Enter the student's details below to predict their final exam score using our optimized XGBoost model.")

st.divider()

# --- Input Form ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Student Details")
    age = st.number_input("Age", min_value=10, max_value=60, value=18, step=1)
    gender = st.selectbox("Gender", options=['Male', 'Female'])
    course = st.selectbox("Course", options=['diploma', 'bca', 'b.sc', 'b.tech', 'bba', 'ba', 'b.com']) # UPDATE THESE IF NEEDED
    internet_access = st.selectbox("Internet Access", options=['Yes', 'No'])

with col2:
    st.subheader("Study Habits")
    study_hours = st.number_input("Study Hours (per week)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    class_attendance = st.number_input("Class Attendance (%)", min_value=0, max_value=100, value=85, step=1)
    study_method = st.selectbox("Study Method", options=['coaching', 'online videos', 'mixed', 'self-study', 'group study']) # UPDATE THESE IF NEEDED
    exam_difficulty = st.selectbox("Exam Difficulty", options=['hard', 'moderate', 'easy']) # UPDATE THESE IF NEEDED

with col3:
    st.subheader("Health & Environment")
    sleep_hours = st.number_input("Sleep Hours (per night)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    sleep_quality = st.selectbox("Sleep Quality", options=['poor', 'average', 'good'])
    facility_rating = st.selectbox("Facility Rating", options=['low', 'medium', 'high'])

st.divider()

# --- Prediction Logic ---
if st.button("Predict Exam Score", type="primary", use_container_width=True):
    try:
        # 1. Map the Ordinal columns exactly as we did in Jupyter
        sleep_map = {'poor': 0, 'average': 1, 'good': 2}
        facility_map = {'low': 0, 'medium': 1, 'high': 2}
        
        # 2. Create a dictionary with ALL expected columns set to 0 initially
        input_dict = {col: 0 for col in expected_columns}
        
        # 3. Populate the numerical and ordinal data
        input_dict['age'] = age
        input_dict['study_hours'] = study_hours
        input_dict['class_attendance'] = class_attendance
        input_dict['sleep_hours'] = sleep_hours
        input_dict['sleep_quality'] = sleep_map[sleep_quality]
        input_dict['facility_rating'] = facility_map[facility_rating]
        
        # 4. Handle the One-Hot Encoded (Nominal) Data
        # We construct the column name it would have created (e.g., 'gender_Male')
        # and if that column exists in our expected list, we flip its 0 to a 1!
        user_categorical_inputs = {
            'gender': gender,
            'course': course,
            'internet_access': internet_access,
            'study_method': study_method,
            'exam_difficulty': exam_difficulty
        }
        
        for category, value in user_categorical_inputs.items():
            expected_dummy_col = f"{category}_{value}"
            if expected_dummy_col in input_dict:
                input_dict[expected_dummy_col] = 1
                
        # 5. Convert to DataFrame (in the exact order the model expects)
        input_df = pd.DataFrame([input_dict])[expected_columns]
        
        # 6. Make Prediction
        prediction = model.predict(input_df)[0]
        
        # 7. Display Result
        st.success(f"### Predicted Exam Score: {prediction:.2f}")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")