
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(page_title="INX Employee Performance Analytics", layout="wide")

st.title("INX Future Inc - Employee Performance Analytics System")

st.markdown("""
This system uses **Machine Learning and HR Analytics** to predict employee performance,
identify employees at risk of underperformance, and support strategic HR decision-making.
""")

# ---------------------------------------
# FILE UPLOAD
# ---------------------------------------

st.sidebar.subheader("Upload Employee Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset uploaded successfully!")
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

# ---------------------------------------
# LOADING MODEL
# ---------------------------------------

model = joblib.load("Rand_Forest_Model.pkl")

# ---------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Overview",
        "Strategic HR Analytics Framework",
        "Performance Prediction",
        "Performance Risk Heatmap",
        "Employee Lifecycle Model",
        "Feature Importance"
    ]
)

# ---------------------------------------
# DATASET OVERVIEW
# ---------------------------------------

if menu == "Dataset Overview":

    st.header("Employee Dataset Overview")

    st.write("Dataset Shape:", data.shape)

    st.dataframe(data.head())

    st.subheader("Performance Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x="PerformanceRating", data=data)
    st.pyplot(fig)


# ------------------------------------------------
# STRATEGIC HR ANALYTICS FRAMEWORK
# ------------------------------------------------
elif menu == "Strategic HR Analytics Framework":

    st.header("Strategic HR Analytics Framework")

    framework = pd.DataFrame({
        "Analytics Level":[
            "Descriptive Analytics",
            "Diagnostic Analytics",
            "Predictive Analytics",
            "Prescriptive Analytics"
        ],
        "Purpose":[
            "Understand workforce trends",
            "Identify causes of performance variation",
            "Predict employee performance outcomes",
            "Recommend strategic HR actions"
        ]
    })

    st.table(framework)

    st.markdown("""
### How INX Uses This Framework

**Descriptive**
Analyze workforce distribution and department performance.

**Diagnostic**
Identify drivers such as salary growth and environment satisfaction.

**Predictive**
Use Random Forest to predict employee performance.

**Prescriptive**
Design HR strategies to improve productivity and engagement.
""")

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------
elif menu == "Feature Importance":

    st.header("Top Drivers of Employee Performance")

    importance = pd.DataFrame({
        "Feature":[
            "EmpLastSalaryHikePercent",
            "EmpEnvironmentSatisfaction",
            "YearsSinceLastPromotion",
            "EmpHourlyRate",
            "EmpJobRole_encoded",
            "YearsWithCurrManager",
            "EmpWorkLifeBalance",
            "EmpDepartment_encoded",
            "NumCompaniesWorked",
            "EmpEducationLevel",
            "EducationBackground_encoded",
            "EmpRelationshipSatisfaction",
            "MaritalStatus",
            "EmpJobInvolvement",
            "BusinessTravelFrequency",
            "Gender_encoded",
            "OverTime_encoded",
            "Attriction_encoded"
        ],
        "Importance":[
            0.276684,0.218195,0.117426,0.049395,0.046875,0.045267,
            0.041277,0.037167,0.028781,0.023207,0.020442,0.019835,
            0.017806,0.017209,0.013197,0.010599,0.009210,0.007429
        ]
    })

    fig, ax = plt.subplots(figsize=(8,10))

    importance.sort_values("Importance").plot(
        kind="barh",
        x="Feature",
        y="Importance",
        ax=ax
    )

    st.pyplot(fig)

    st.markdown("""
The most influential factors affecting employee performance are:

• Salary Hike Percentage  
• Environment Satisfaction  
• Years Since Last Promotion

These variables represent **motivation, recognition, and career growth drivers**.
""")



# ---------------------------------------
# PERFORMANCE PREDICTION
# ---------------------------------------


elif menu == "Performance Prediction":

    st.header("Employee Performance Prediction")

    st.markdown("Enter employee details to predict performance level.")

    gender_map = {
    0: "Female",
    1: "Male"}
    
    overtime_map = {
    0: "No",
    1: "Yes"}
    
    attrition_map = {
    0: "No",
    1: "Yes"}
    
    marital_map = {
    0: "Single",
    1: "Married",
    2: "Divorced"}

    department_map = {
    0: "Sales",
    1: "Development",
    2: "Research & Development",
    3: "Human Resources",
    4: "Finance",
    5: "Data Science"}

    jobrole_map = {
    0: "Manager",
    1: "Developer",
    2: "Sales Executive",
    3: "Research Scientist",
    4: "HR Specialist",
    5: "Finance Analyst",
    6: "Data Scientist",
    7: "Business Analyst",
    8: "Delivery Manager",
    9: "Developer",
    10: "Finance Manager",
    11: "Healthcare Representative",
    12: "Human Resources",
    13: "Laboratory Technician",
    14: "Manager R&D",
    15: "Manufacturing Director",
    16: "Research Director",
    17: "Sales Representative",
    18: "Senior Developer",
    19: "Senior Manager R&D",
    20: "Technical Architect",
    21: "Technical Lead"
    }

    travel_map = {
    0: "Non-Travel",
    1: "Travel Rarely",
    2: "Travel Frequently"}

    education_background_map = {
    0: "Life Sciences",
    1: "Medical",
    2: "Marketing",
    3: "Technical Degree",
    4: " Human Resources",
    5: "Other"}

    col1, col2, col3 = st.columns(3)

    with col1:
        salary_hike = st.slider("Salary Hike %", 0, 30, 12)
        env_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 2)
        years_promotion = st.slider("Years Since Last Promotion", 0, 15, 10)
        hourly_rate = st.slider("Hourly Rate", 30, 100, 30)
        job_role_name = st.selectbox("Job Role", list(jobrole_map.values()))
        job_role = list(jobrole_map.keys())[list(jobrole_map.values()).index(job_role_name)]

    with col2:
        years_manager = st.slider("Years With Current Manager", 0, 20, 5)
        work_life = st.slider("Work Life Balance (1-4)", 1, 4, 2)
        department = st.selectbox(" Department",list(department_map.values()) )
        department = list(department_map.keys())[list(department_map.values()).index(department)]
        companies = st.slider("Num Companies Worked", 0, 10, 2)
        education = st.slider("Education Level", 1, 5, 3)

    with col3:
        edu_background = st.selectbox("Education Background", list(education_background_map.values()))
        edu_background = list(education_background_map.keys())[list(education_background_map.values()).index(edu_background)]
        relationship = st.slider("Relationship Satisfaction", 1, 4, 3)
        marital = st.selectbox("Marital Status", list(marital_map.values()))
        marital = list(marital_map.keys())[list(marital_map.values()).index(marital)]
        job_involve = st.slider("Job Involvement", 1, 4, 3)
        travel = st.selectbox("Business Travel Frequency", list(travel_map.values()))
        travel = list(travel_map.keys())[list(travel_map.values()).index(travel)]
        gender = st.selectbox("Gender", list(gender_map.values()))
        gender = list(gender_map.keys())[list(gender_map.values()).index(gender)]
        overtime_name = st.selectbox("Overtime", list(overtime_map.values()))
        overtime = list(overtime_map.keys())[list(overtime_map.values()).index(overtime_name)]
        attritions = st.selectbox("Attrition", ["No", "Yes"])
        attrition = list(attrition_map.keys())[list(attrition_map.values()).index(attritions)]

    input_data = pd.DataFrame({
        "EmpLastSalaryHikePercent":[salary_hike],
        "EmpEnvironmentSatisfaction":[env_satisfaction],
        "YearsSinceLastPromotion":[years_promotion],
        "EmpHourlyRate":[hourly_rate],
        "EmpJobRole_encoded":[job_role],
        "YearsWithCurrManager":[years_manager],
        "EmpWorkLifeBalance":[work_life],
        "EmpDepartment_encoded":[department],
        "NumCompaniesWorked":[companies],
        "EmpEducationLevel":[education],
        "EducationBackground_encoded":[edu_background],
        "EmpRelationshipSatisfaction":[relationship],
        "MaritalStatus":[marital],
        "EmpJobInvolvement":[job_involve],
        "BusinessTravelFrequency":[travel],
        "Gender_encoded":[gender],
        "OverTime_encoded":[overtime],
        "Attriction_encoded":[attrition]
    })

    if st.button("Predict Performance"):

        prediction = model.predict(input_data)[0]

        if prediction == 0:
            st.error("Predicted Performance: Low Performer")
        elif prediction == 1:
            st.warning("Predicted Performance: Average Performer")
        else:
            st.success("Predicted Performance: High Performer")

# ------------------------------------------------
# PERFORMANCE RISK HEATMAP
# ------------------------------------------------
elif menu == "Performance Risk Heatmap":

    st.header("Employee Underperformance Risk Heatmap")

    st.markdown("""
This heatmap highlights employees most likely to **underperform** based on key drivers:

• Salary Hike  
• Environment Satisfaction  
• Promotion Delay
""")

    risk_data = data.copy()

    risk_data["RiskScore"] = (
        (30 - risk_data["EmpLastSalaryHikePercent"]) * 0.4 +
        (4 - risk_data["EmpEnvironmentSatisfaction"]) * 0.3 +
        risk_data["YearsSinceLastPromotion"] * 0.3
    )

    pivot = pd.pivot_table(
        risk_data,
        values="RiskScore",
        index="EmpEnvironmentSatisfaction",
        columns="YearsSinceLastPromotion",
        aggfunc=np.mean
    )

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(pivot, cmap="Reds", annot=True)

    st.pyplot(fig)

    st.markdown("""
Higher values indicate **greater risk of employee underperformance**.

HR managers should prioritize employees in the **high-risk zones** for engagement programs.
""")

# ------------------------------------------------
# EMPLOYEE PERFORMANCE LIFECYCLE
# ------------------------------------------------
elif menu == "Employee Lifecycle Model":

    st.header("Employee Performance Lifecycle Model")

    st.markdown("""
Employees generally move through four performance stages in organizations.
""")

    lifecycle = pd.DataFrame({
        "Stage":["Onboarding","Growth","Plateau","Decline"],
        "Performance":[50,85,70,40]
    })

    fig, ax = plt.subplots()

    ax.plot(lifecycle["Stage"], lifecycle["Performance"], marker="o", linewidth=3)

    ax.set_ylabel("Performance Level")

    ax.set_title("Employee Performance Lifecycle")

    st.pyplot(fig)

    st.markdown("""
### Lifecycle Stages

**Onboarding**
Employees are learning and adapting.

**Growth**
Employees develop skills and reach high productivity.

**Plateau**
Performance stabilizes and motivation may slow.

**Decline**
Without promotions or incentives, engagement drops.
""")



