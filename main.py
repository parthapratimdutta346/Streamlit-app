import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "best_xgb_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "feature_columns.pkl"


NUMERICAL_COLS = [
	"cgpa",
	"10th mark",
	"12th mark",
	"backlogs",
	"attendance_percentage",
	"projects_completed",
	"internships",
	"technical_skill",
	"communication",
	"aptitude",
	"hackathons",
	"certifications",
]

CATEGORICAL_COLS = [
	"gender",
	"branch",
	"extracurriculars",
]


@st.cache_resource
def load_assets():
	if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH)):
		return None, None, None

	model = joblib.load(MODEL_PATH)
	scaler = joblib.load(SCALER_PATH)
	feature_columns = joblib.load(FEATURES_PATH)
	return model, scaler, feature_columns


def preprocess_input(raw_data, scaler, feature_columns):
	input_df = pd.DataFrame([raw_data])

	# Drop identifiers if present.
	input_df = input_df.drop(columns=["Student_ID"], errors="ignore")

	# One-hot encode categorical columns.
	cols_to_dummy = [col for col in CATEGORICAL_COLS if col in input_df.columns]
	input_df_encoded = pd.get_dummies(input_df, columns=cols_to_dummy, drop_first=True)

	# Ensure numerical columns exist before scaling.
	for col in NUMERICAL_COLS:
		if col not in input_df_encoded.columns:
			input_df_encoded[col] = 0.0

	input_df_encoded[NUMERICAL_COLS] = scaler.transform(input_df_encoded[NUMERICAL_COLS])

	# Align to training feature columns.
	processed = pd.DataFrame(columns=feature_columns)
	for col in feature_columns:
		processed[col] = input_df_encoded[col] if col in input_df_encoded.columns else 0

	return processed


def predict(model, processed_input):
	prediction = model.predict(processed_input)[0]

	if hasattr(model, "predict_proba"):
		probability = model.predict_proba(processed_input)[:, 1][0]
	elif hasattr(model, "decision_function"):
		score = model.decision_function(processed_input)[0]
		probability = 1 / (1 + pow(2.718281828, -score))
	else:
		probability = 0.0

	label = "Placed" if int(prediction) == 1 else "Not Placed"
	return label, float(probability)


st.set_page_config(page_title="Placement Predictor", layout="wide")

st.markdown(
	"""
	<style>
	body {
		background-color: #f5f1ea;
	}
	.main {
		background-color: #f5f1ea;
	}
	h1, h2, h3, p, label {
		color: #5c4b3b;
	}
	.stButton > button {
		background-color: #8b5e3c;
		color: white;
		border: none;
		border-radius: 12px;
		padding: 12px 28px;
		font-size: 16px;
		font-weight: 600;
		cursor: pointer;
	}
	.stButton > button:hover {
		background-color: #a47148;
		color: white;
	}
	</style>
	""",
	unsafe_allow_html=True,
)

st.title("Student Placement Prediction")
st.write("Enter student details to predict placement likelihood.")

model, scaler, feature_columns = load_assets()

if model is None or scaler is None or feature_columns is None:
	st.error(
		"Model assets not found. Please ensure best_xgb_model.pkl, scaler.pkl, and feature_columns.pkl exist in this folder."
	)
	st.stop()

with st.form("prediction_form"):
	col1, col2, col3 = st.columns(3)

	with col1:
		gender = st.selectbox("Gender", ["Male", "Female", "Other"])
		branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "EEE", "Mech", "Civil", "Other"])
		extracurriculars = st.selectbox("Extracurriculars", ["Low", "Medium", "High"])

	with col2:
		cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
		tenth_mark = st.number_input("10th Mark", min_value=0.0, max_value=100.0, value=85.0, step=0.5)
		twelfth_mark = st.number_input("12th Mark", min_value=0.0, max_value=100.0, value=78.0, step=0.5)
		attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=90.0, step=0.5)

	with col3:
		backlogs = st.number_input("Backlogs", min_value=0, max_value=20, value=0, step=1)
		projects = st.number_input("Projects Completed", min_value=0, max_value=20, value=1, step=1)
		internships = st.number_input("Internships", min_value=0, max_value=10, value=1, step=1)
		technical_skill = st.slider("Technical Skill", 0, 5, 3)
		communication = st.slider("Communication", 0, 5, 3)
		aptitude = st.slider("Aptitude", 0, 100, 50)
		hackathons = st.number_input("Hackathons", min_value=0, max_value=10, value=1, step=1)
		certifications = st.number_input("Certifications", min_value=0, max_value=10, value=2, step=1)

	submitted = st.form_submit_button("Predict")

def compute_performance_score(cgpa_value, attendance_value, backlogs_value, certifications_value, projects_value, technical_value, internships_value, aptitude_value, communication_value):
	score = 0

	if cgpa_value >= 8.5:
		score += 5
	elif cgpa_value >= 7.5:
		score += 4
	elif cgpa_value >= 6.5:
		score += 3
	else:
		score += 1

	if attendance_value >= 85:
		score += 3
	elif attendance_value >= 75:
		score += 2
	else:
		score += 1

	if backlogs_value == 0:
		score += 4
	elif backlogs_value == 1:
		score += 2
	else:
		score += 0

	if certifications_value >= 2:
		score += 3
	elif certifications_value == 1:
		score += 2
	else:
		score += 1

	if projects_value >= 2:
		score += 3
	elif projects_value == 1:
		score += 2
	else:
		score += 1

	if technical_value > 3:
		score += 2
	else:
		score += 1

	if internships_value > 0:
		score += 2
	else:
		score += 1

	if aptitude_value >= 70:
		score += 3
	elif aptitude_value >= 50:
		score += 2
	else:
		score += 1

	if communication_value >= 4:
		score += 3
	elif communication_value >= 2:
		score += 2
	else:
		score += 1

	return score


def performance_category(score_value):
	if score_value >= 22:
		return "Excellent"
	if score_value >= 15:
		return "Moderate"
	return "Poor"


if submitted:
	raw_input = {
		"gender": gender,
		"branch": branch,
		"extracurriculars": extracurriculars,
		"cgpa": cgpa,
		"10th mark": tenth_mark,
		"12th mark": twelfth_mark,
		"attendance_percentage": attendance,
		"backlogs": backlogs,
		"projects_completed": projects,
		"internships": internships,
		"technical_skill": technical_skill,
		"communication": communication,
		"aptitude": aptitude,
		"hackathons": hackathons,
		"certifications": certifications,
	}

	processed = preprocess_input(raw_input, scaler, feature_columns)
	label, probability = predict(model, processed)

	score = compute_performance_score(
		cgpa,
		attendance,
		backlogs,
		certifications,
		projects,
		technical_skill,
		internships,
		aptitude,
		communication,
	)
	category = performance_category(score)

	st.subheader("Placement Status")
	if label == "Placed":
		st.success(label)
	else:
		st.error(label)

	st.subheader("Performance Category")
	if category == "Excellent":
		st.success(category)
	elif category == "Moderate":
		st.warning(category)
	else:
		st.error(category)

	metrics_left, metrics_right = st.columns(2)
	with metrics_left:
		st.metric("Placement Probability", f"{probability * 100:.2f}%")
	with metrics_right:
		st.metric("Performance Score", f"{score}")

	st.progress(min(max(probability, 0.0), 1.0))

	st.subheader("Academic Breakdown")
	st.progress(min(int(cgpa * 10), 100))
	st.write(f"CGPA: {cgpa:.2f}")
	st.progress(int(attendance))
	st.write(f"Attendance: {attendance:.1f}")
	st.progress(int(aptitude))
	st.write(f"Aptitude: {aptitude}")
	st.progress(min(projects * 20, 100))
	st.write(f"Projects: {projects}")
	st.progress(min(technical_skill * 20, 100))
	st.write(f"Technical Skill: {technical_skill}")

