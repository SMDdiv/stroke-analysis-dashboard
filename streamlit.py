import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Sidebar Styling :
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 300px;
        max-width: 300px;
        background-color: #0a0a0f;
    }
    section[data-testid="stSidebar"] {
        padding: 2.5rem 2rem;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 1.5rem;
        font-weight: bold;
        color: #dddddd;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stSidebar"] label {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.15rem;
        font-weight: 500;
        padding: 0.6rem 0.75rem;
        color: #cccccc;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stSidebar"] label:hover {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
        cursor: pointer;
    }
    [data-testid="stSidebar"] input:checked ~ div > span {
        background-color: #14B8A6 !important;
        border: 2px solid #444444;
    }
    div[data-testid="stSidebar"] input[type="radio"] {
        transform: scale(1.2);
        margin-right: 6px;
    }
    div[data-testid="stSidebar"] input[type="radio"] + div {
        display: flex;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Stroke Prediction EDA Dashboard",
                   layout="wide",
                   page_icon="\U0001F4CA")

st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to", ["Description", "Model", "Dashboard"])

@st.cache_data
def load_data(path: str = r"healthcare-dataset-stroke-data.csv") -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other'].copy()

    def glucose_category(value):
        if value < 70: return 'Low'
        elif value <= 140: return 'Normal'
        else: return 'High'
    df['glucose_category'] = df['avg_glucose_level'].apply(glucose_category)

    def bmi_category(bmi):
        if bmi < 18.5: return 'Underweight'
        elif bmi < 25: return 'Normal'
        elif bmi < 30: return 'Overweight'
        else: return 'Obese'
    df['bmi_category'] = df['bmi'].apply(bmi_category)

    bins = [0, 30, 45, 60, 75, 100]
    labels = ['0-30', '31-45', '46-60', '61-75', '76+']
    df['age_bucket'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

    df['combo'] = df['hypertension'].astype(str) + df['heart_disease'].astype(str)
    combo_map = {
        '00': 'No HTN & No Heart Disease',
        '10': 'Hypertension Only',
        '01': 'Heart Disease Only',
        '11': 'Both Conditions'
    }
    df['combo'] = df['combo'].map(combo_map)

    return df

if page == "Description":
    st.title("\U0001F4C4 Project Description: Stroke Risk Analysis")

    st.markdown("""
    ### \U0001F9E0 What are we solving?
    Stroke is a leading cause of disability and death worldwide. Many strokes are preventable if early risk factors are identified.

    ### \U0001F3AF Our Target Audience
    - Healthcare providers
    - Policy makers
    - Data scientists in health tech

    ### ✅ What makes this dashboard effective?
    - Clean, real-time filtering
    - Easy-to-understand metrics
    - Predictive insights into stroke risks
    - Visual breakdown by age, gender, glucose, BMI

    ### \U0001F4CA Dataset Overview
    The dataset contains demographic and health information of individuals and whether they experienced a stroke.
    """)

    df = load_data(r"healthcare-dataset-stroke-data.csv")
    if df is not None:
        df = preprocess_data(df)

        st.markdown("### \U0001F522 Basic Statistics for All Columns")
        st.dataframe(df.describe(include='all').T)

        st.markdown("### \U0001F4C8 L1 & L2 Norm Analysis")
        features = ['age', 'avg_glucose_level', 'bmi']
        X = df[features].copy()
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        feature_importance = np.array([0.2, 0.3, 0.5])
        l1_norm = np.sum(np.abs(feature_importance))
        l2_norm = np.sqrt(np.sum(feature_importance ** 2))

        st.write("L1 Norm (Sum of absolute values):", l1_norm)
        st.write("L2 Norm (Square root of sum of squares):", l2_norm)

        st.markdown("""
        - **L1 Norm** reflects the overall contribution of features. It helps in feature selection by promoting sparsity (used in Lasso Regression).
        - **L2 Norm** emphasizes the magnitude of large values and is used for feature smoothing and stability (used in Ridge Regression).
        """)

        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
        fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importances (L1 & L2 Norms Approximation)')
        st.plotly_chart(fig)

elif page == "Model":

    # Load the trained model
    model = joblib.load('model.pkl')

    st.title("🔍 Stroke Prediction System")

    # Section: User Input
    st.header("🧑‍⚕️ Enter Patient Data")

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        # Age input (slider)
        age = st.slider("Age (years)", 1, 100, 30)
        # Weight input (number)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        # Hypertension input (radio button)
        hypertension = st.radio("Hypertension?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        # Heart disease input (radio button)
        heart_disease = st.radio("Heart Disease?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        # Smoking status input (radio button)
        is_smoker = st.radio("Smoker?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col2:
        # Height input (number)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
        # Average glucose level input (number)
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=90.0)
        # Gender input (dropdown)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        # Marital status input (dropdown)
        ever_married = st.selectbox("Ever Married?", ['Yes', 'No'])
        # Residence type input (dropdown)
        residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])

    # Calculate BMI automatically
    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 2)

    # Function to classify life stage based on age
    def classify_life_stage(age):
        if age < 1:
            return 'Infant'
        elif age <= 3:
            return 'Toddler'
        elif age <= 12:
            return 'Child'
        elif age <= 15:
            return 'Early Adolescent'
        elif age <= 19:
            return 'Late Adolescent'
        elif age <= 35:
            return 'Early Youth'
        elif age <= 50:
            return 'Mid Youth'
        elif age <= 65:
            return 'Early Adulthood'
        elif age <= 80:
            return 'Late Adulthood'
        elif age <= 90:
            return 'Elderly'
        else:
            return 'Centenarian'

    # Get life stage for the input age
    life_stage = classify_life_stage(age)

    # Function to classify BMI
    def classify_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'

    # Get BMI category for the calculated BMI
    bmi_category = classify_bmi(bmi)

    # Function to classify glucose level
    def classify_glucose(glucose):
        if glucose < 70:
            return 'Hypoglycemia'
        elif glucose <= 99:
            return 'Normal'
        elif glucose <= 125:
            return 'Prediabetes'
        else:
            return 'Diabetes'

    # Get glucose category for the input glucose level
    glucose_category = classify_glucose(avg_glucose_level)

    # When the user clicks the prediction button
    if st.button("🔮 Predict Stroke"):
        # Prepare user input as a DataFrame for the model
        user_input = pd.DataFrame([{
            'age': age,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'is_smoker': is_smoker,
            'gender': gender,
            'ever_married': ever_married,
            'work_type': 'Private',  # Default work type
            'Residence_type': residence_type,
            'life_stage': life_stage,
            'bmi_category': bmi_category,
            'glucose_level_category': glucose_category
        }])

        # Make prediction using the loaded model
        prediction = model.predict(user_input)[0]
        probability = model.predict_proba(user_input)[0][1]

        # Display the prediction result
        st.subheader("🔎 Result:")
        if prediction == 1:
            st.error(f"⚠️ High risk of stroke. Probability: {probability * 100:.2f}%")
        else:
            st.success(f"✅ No significant risk. Probability: {probability * 100:.2f}%")

        st.markdown("---")
        st.markdown("**Your Data:**")
        st.dataframe(user_input)

    # === Visualizations Section ===
    # Load test set and predictions
    if os.path.exists(r'X_test.csv') and os.path.exists(r'y_test.csv') and os.path.exists(r'y_pred_proba.csv'):
        X_test_vis = pd.read_csv(r'X_test.csv')
        y_test_vis = pd.read_csv(r'y_test.csv')['y_test']
        y_pred_proba_vis = pd.read_csv(r'y_pred_proba.csv')
        y_pred_vis = y_pred_proba_vis['y_pred']
        y_proba_vis = y_pred_proba_vis['y_proba']

        # Feature Importance (reuse model pipeline)
        onehot_feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out([
            'hypertension', 'heart_disease', 'is_smoker',
            'gender', 'ever_married', 'work_type', 'Residence_type',
            'life_stage', 'bmi_category', 'glucose_level_category'
        ])
        numeric_features = ['age', 'avg_glucose_level', 'bmi']
        all_features = list(numeric_features) + list(onehot_feature_names)
        importances = model.named_steps['classifier'].feature_importances_
        importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig1 = px.bar(
            importance_df.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Important Features for Stroke Prediction'
        )
        fig1.update_layout(yaxis={'categoryorder':'total ascending'})
        st.subheader('Feature Importance')
        st.plotly_chart(fig1)

        # Confusion Matrix
        cm = confusion_matrix(y_test_vis, y_pred_vis)
        labels = ['No Stroke', 'Stroke']
        fig2 = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True,
            annotation_text=[[str(cell) for cell in row] for row in cm]
        )
        fig2.update_layout(title='Confusion Matrix')
        st.subheader('Confusion Matrix')
        st.plotly_chart(fig2)

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test_vis, y_proba_vis)
        roc_auc = auc(fpr, tpr)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.2f})'))
        fig3.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig3.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        st.subheader('ROC Curve')
        st.plotly_chart(fig3)



elif page == "Dashboard":
    st.title("Stroke Prediction EDA Dashboard")
    st.caption("Visual insights into stroke risk based on demographic and health factors.")

    df = load_data("healthcare-dataset-stroke-data.csv")
    if df is None:
        st.warning("Dataset not found. Please upload `healthcare-dataset-stroke-data.csv`.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

    if df is not None:
        df = preprocess_data(df)

        st.sidebar.subheader("Filter Data")

        # ✅ Reset logic: fully clear session state keys
        if st.sidebar.button("Reset Filters"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        genders = df['gender'].unique().tolist()
        gender_filter = st.sidebar.multiselect("Select Gender", options=genders, default=st.session_state.get('gender_filter', genders))
        st.session_state['gender_filter'] = gender_filter

        age_min, age_max = int(df['age'].min()), int(df['age'].max())
        age_filter = st.sidebar.slider("Select Age Range", min_value=age_min, max_value=age_max,
                                       value=st.session_state.get('age_filter', (age_min, age_max)))
        st.session_state['age_filter'] = age_filter

        marital_options = df['ever_married'].unique().tolist()
        marital_filter = st.sidebar.multiselect("Select Marital Status", options=marital_options,
                                               default=st.session_state.get('marital_filter', marital_options))
        st.session_state['marital_filter'] = marital_filter

        work_options = df['work_type'].unique().tolist()
        work_filter = st.sidebar.multiselect("Select Work Type", options=work_options,
                                            default=st.session_state.get('work_filter', work_options))
        st.session_state['work_filter'] = work_filter

        residence_options = ['Both'] + df['Residence_type'].unique().tolist()
        residence_filter = st.sidebar.radio("Residence Type", options=residence_options,
                                           index=residence_options.index(st.session_state.get('residence_filter', 'Both')))
        st.session_state['residence_filter'] = residence_filter

        # Filtering logic
        filtered_df = df.copy()
        if residence_filter != 'Both':
            filtered_df = filtered_df[filtered_df['Residence_type'] == residence_filter]

        filtered_df = filtered_df[
            (filtered_df['gender'].isin(gender_filter)) &
            (filtered_df['age'] >= age_filter[0]) & (filtered_df['age'] <= age_filter[1]) &
            (filtered_df['ever_married'].isin(marital_filter)) &
            (filtered_df['work_type'].isin(work_filter))
        ]

        total_records = len(filtered_df)
        total_strokes = int(filtered_df['stroke'].sum())
        stroke_rate_overall = (total_strokes / total_records) * 100 if total_records > 0 else 0

        age_stroke = filtered_df.groupby('age_bucket', observed=False)['stroke'].mean().reset_index()
        age_stroke['stroke_percent'] = age_stroke['stroke'] * 100
        highest_age = age_stroke.loc[age_stroke['stroke'].idxmax()]['age_bucket']

        bmi_stroke = filtered_df.groupby('bmi_category', observed=False)['stroke'].mean().reset_index()
        bmi_stroke['stroke_percent'] = bmi_stroke['stroke'] * 100
        highest_bmi = bmi_stroke.loc[bmi_stroke['stroke'].idxmax()]['bmi_category']

        combo_stroke = filtered_df.groupby('combo', observed=False)['stroke'].mean().reset_index()
        combo_stroke['stroke_percent'] = combo_stroke['stroke'] * 100
        highest_combo = combo_stroke.loc[combo_stroke['stroke'].idxmax()]['combo']

        glucose_stroke = filtered_df.groupby('glucose_category', observed=False)['stroke'].mean().reset_index()
        glucose_stroke['stroke_percent'] = glucose_stroke['stroke'] * 100

        gender_stroke = filtered_df.groupby('gender', observed=False)['stroke'].mean().reset_index()
        gender_stroke['stroke_percent'] = gender_stroke['stroke'] * 100

        st.subheader("Key Metrics and Insights")
        cols1 = st.columns(3)
        cols1[0].metric("Total Participants", f"{total_records}")
        cols1[1].metric("Stroke Rate (Overall)", f"{stroke_rate_overall:.2f}%")
        cols1[2].metric("Total Stroke Events", f"{total_strokes}")

        cols2 = st.columns(3)
        cols2[0].metric("Highest-Risk Age Group", highest_age)
        cols2[1].metric("Highest-Risk BMI Category", highest_bmi)
        cols2[2].metric("Highest-Risk Condition", highest_combo)

        st.markdown("---")

        color_palette = px.colors.sequential.Teal

        with st.container():
            st.subheader("Stroke Rate by Age Group")
            age_stroke = filtered_df.groupby('age_bucket', observed=False)['stroke'].mean().reset_index()
            age_stroke['stroke_percent'] = age_stroke['stroke'] * 100
            sorted_age = age_stroke.sort_values('stroke_percent', ascending=False)
            fig_age = px.bar(
                sorted_age,
                x='age_bucket',
                y='stroke_percent',
                color='age_bucket',
                labels={'stroke_percent': 'Stroke Rate (%)', 'age_bucket': 'Age Group'},
                text=sorted_age['stroke_percent'].round(1),
                color_discrete_sequence=color_palette
            )
            fig_age.update_layout(template='plotly_dark', showlegend=False)
            fig_age.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
            st.plotly_chart(fig_age, use_container_width=True)
            top_age = sorted_age.iloc[0]['age_bucket']
            st.caption(f"The age group {top_age} has the highest stroke rate")

        # Chart 2: Stroke Rate by Gender
        # Compares stroke rate across genders using a donut-style pie chart
        with st.container():
            st.subheader("Stroke Rate by Gender")
            fig_gender = px.pie(
                gender_stroke,
                names='gender',
                values='stroke_percent',
                color='gender',
                hole=0.3,
                color_discrete_sequence=color_palette
            )
            fig_gender.update_layout(template='plotly_dark', showlegend=True)
            st.plotly_chart(fig_gender, use_container_width=True)
            top_gender = gender_stroke.sort_values('stroke_percent', ascending=False).iloc[0]['gender']
            st.caption(f"{top_gender} participants have the highest stroke rate")

        # Chart 3: Stroke Rate by Work Type
        # Helps uncover whether lifestyle or job-related stress influences stroke risk
        with st.container():
            st.subheader("Stroke Rate by Work Type")
            work_stroke = filtered_df.groupby('work_type', observed=False)['stroke'].mean().reset_index()
            work_stroke['stroke_percent'] = work_stroke['stroke'] * 100
            sorted_work = work_stroke.sort_values('stroke_percent', ascending=False)
            fig_work = px.bar(
                sorted_work,
                y='work_type',
                x='stroke_percent',
                orientation='h',
                color='work_type',
                labels={'stroke_percent': 'Stroke Rate (%)', 'work_type': 'Work Type'},
                text=sorted_work['stroke_percent'].round(1),
                color_discrete_sequence=color_palette
            )
            fig_work.update_layout(template='plotly_dark', showlegend=False)
            fig_work.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
            st.plotly_chart(fig_work, use_container_width=True)
            top_work = sorted_work.iloc[0]['work_type']
            st.caption(f"Stroke risk is highest among people working in {top_work}")

        # Chart 4: Stroke Count by Smoking Status
        # Compares total number of stroke cases across smoking categories
        with st.container():
            st.subheader("Stroke Count by Smoking Status")
            smoke_counts = filtered_df.groupby('smoking_status', observed=False)['stroke'].sum().reset_index()
            smoke_counts = smoke_counts.rename(columns={'stroke': 'stroke_count'})
            sorted_smoke = smoke_counts.sort_values('stroke_count', ascending=False)
            fig_smoke = px.bar(
                sorted_smoke,
                x='smoking_status',
                y='stroke_count',
                color='smoking_status',
                labels={'stroke_count': 'Stroke Events', 'smoking_status': 'Smoking Status'},
                text=sorted_smoke['stroke_count'],
                color_discrete_sequence=color_palette
            )
            fig_smoke.update_layout(template='plotly_dark', showlegend=False)
            fig_smoke.update_traces(textposition='outside', texttemplate='%{text}')
            st.plotly_chart(fig_smoke, use_container_width=True)
            top_smoke = sorted_smoke.iloc[0]['smoking_status']
            st.caption(f"The group with the most stroke cases is: {top_smoke}")

        # Chart 5: Stroke Rate by Marital Status
        # Because marital status is just Yes/No, a pie makes it easier to compare the two groups
        with st.container():
            st.subheader("Stroke Rate by Marital Status")
            # Calculate the average stroke rate for each marital status
            marital_stroke = filtered_df.groupby('ever_married', observed=False)['stroke'].mean().reset_index()
            marital_stroke['stroke_percent'] = marital_stroke['stroke'] * 100
            # Build a simple donut-style pie chart
            fig_marital = px.pie(
                marital_stroke,
                names='ever_married',
                values='stroke_percent',
                color='ever_married',
                hole=0.3,
                labels={'stroke_percent': 'Stroke Rate (%)', 'ever_married': 'Marital Status'},
                color_discrete_sequence=color_palette
            )
            # Increase the legend text size for better readability
            fig_marital.update_layout(template='plotly_dark', showlegend=True, legend=dict(font=dict(size=14)))
            st.plotly_chart(fig_marital, use_container_width=True)
            top_marital = marital_stroke.sort_values('stroke_percent', ascending=False).iloc[0]['ever_married']
            # Give a concise insight without the awkward colon
            st.caption(f"Highest stroke rate appears among people who are {top_marital}")
            
        # Chart 6: Stroke Rate by Glucose Level Band
        # Categorizes glucose into bands to see how sugar levels correlate with stroke risk	
        with st.container():
            st.subheader("Stroke Rate by Glucose Level Band")

            # Create glucose bands (Low, Normal, High, Very High)
            filtered_df["glucose_band"] = pd.cut(
                filtered_df["avg_glucose_level"],
                bins=[0, 100, 140, 200, 300],
                labels=["Low", "Normal", "High", "Very High"]
            )

            # Compute stroke rate for each band
            glucose_band_stroke = filtered_df.groupby("glucose_band", observed=False)["stroke"].mean().reset_index()
            glucose_band_stroke["stroke_percent"] = glucose_band_stroke["stroke"] * 100

            # Sort by stroke rate descending
            sorted_glu_band = glucose_band_stroke.sort_values(by="stroke_percent", ascending=False)

            # Plot bar chart
            fig_glu_band = px.bar(
                sorted_glu_band,
                x="glucose_band",
                y="stroke_percent",
                labels={"stroke_percent": "Stroke Rate (%)", "glucose_band": "Glucose Band"},
                text=sorted_glu_band["stroke_percent"].round(1),
                color="glucose_band",
                color_discrete_sequence=color_palette
            )

            fig_glu_band.update_traces(textposition="outside", texttemplate='%{text:.1f}%')
            fig_glu_band.update_layout(
                template='plotly_dark',
                showlegend=False
            )

            st.plotly_chart(fig_glu_band, use_container_width=True)

            top_band = sorted_glu_band.iloc[0]['glucose_band']
            st.caption(f"The {top_band} glucose group has the highest stroke rate in this banded view")

        # Chart 7: Stroke Rate by BMI Category
        # Compares how underweight, normal, overweight, obese groups are affected
        with st.container():
            st.subheader("Stroke Rate by BMI Category")
            bmi_stroke = filtered_df.groupby('bmi_category', observed=False)['stroke'].mean().reset_index()
            bmi_stroke['stroke_percent'] = bmi_stroke['stroke'] * 100
            sorted_bmi = bmi_stroke.sort_values('stroke_percent', ascending=False)
            fig_bmi = px.bar(
                sorted_bmi,
                x='bmi_category',
                y='stroke_percent',
                color='bmi_category',
                labels={'stroke_percent': 'Stroke Rate (%)', 'bmi_category': 'BMI Category'},
                text=sorted_bmi['stroke_percent'].round(1),
                color_discrete_sequence=color_palette
            )
            fig_bmi.update_layout(template='plotly_dark', showlegend=False)
            fig_bmi.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
            st.plotly_chart(fig_bmi, use_container_width=True)
            # Highlight which BMI group has the highest stroke rate
            top_bmi = sorted_bmi.iloc[0]['bmi_category']
            st.caption(f"Highest stroke rate appears among {top_bmi} BMI group")

        # Chart 8: Stroke Rate by Life Stage and Gender
        # Compares different age groups and genders to see which combination has the highest risk
        with st.container():
            st.subheader("Stroke Rate by Life Stage and Gender")
            life_stage = filtered_df.groupby(['age_bucket', 'gender'], observed=False)['stroke'].mean().reset_index()
            life_stage['stroke_percent'] = life_stage['stroke'] * 100
            # Order age_bucket by highest stroke rate across genders (descending)
            order_life = (
                life_stage.groupby('age_bucket')['stroke_percent']
                .max()
                .sort_values(ascending=False)
                .index.tolist()
            )
            fig_life = px.bar(
                life_stage,
                x='age_bucket',
                y='stroke_percent',
                color='gender',
                barmode='group',
                labels={'stroke_percent': 'Stroke Rate (%)', 'age_bucket': 'Age Group', 'gender': 'Gender'},
                text=life_stage['stroke_percent'].round(1),
                category_orders={'age_bucket': order_life},
                color_discrete_sequence=color_palette
            )
            fig_life.update_layout(template='plotly_dark', showlegend=True)
            fig_life.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
            st.plotly_chart(fig_life, use_container_width=True)
            # Determine highest risk combination
            max_idx = life_stage['stroke_percent'].idxmax()
            top_age_group = life_stage.loc[max_idx, 'age_bucket']
            top_gender_group = life_stage.loc[max_idx, 'gender']
            st.caption(f"The highest rate is seen in {top_gender_group} in the {top_age_group} age group")
        # Chart 9: Stroke Rate by Hypertension and Heart Disease Combination
        st.subheader("Stroke Rate by Hypertension and Heart Disease Combination")
        combo_stroke = combo_stroke.sort_values('stroke_percent', ascending=False)
        fig_combo = px.bar(
            combo_stroke,
            x='combo',
            y='stroke_percent',
            color='combo',
            text='stroke_percent',
            labels={'stroke_percent': 'Stroke Rate (%)', 'combo': 'Condition Combination'},
            color_discrete_sequence=color_palette
        )
        fig_combo.update_layout(template='plotly_dark', showlegend=False)
        fig_combo.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
        st.plotly_chart(fig_combo, use_container_width=True)
        # Provide a simple insight for the combo chart
        top_combo = combo_stroke.iloc[0]['combo']
        st.caption(f"The highest stroke rate occurs in the {top_combo} group")

        # Chart 10: Stroke Rate by BMI and Glucose Level (Heatmap)
        # Multivariable chart showing interaction between obesity and blood sugar
        st.subheader("Stroke Rate by BMI and Glucose Level")
        heatmap_data = filtered_df.groupby(['bmi_category', 'glucose_category'], observed=False)['stroke'].mean().reset_index()
        heatmap_data['stroke_percent'] = heatmap_data['stroke'] * 100
        heatmap_pivot = heatmap_data.pivot(index='glucose_category', columns='bmi_category', values='stroke_percent')
        fig_heatmap = px.imshow(heatmap_pivot, text_auto=True, color_continuous_scale='Teal')
        fig_heatmap.update_layout(template='plotly_dark', title='Stroke Rate (%) by BMI and Glucose Category')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        # Provide an insight for the heatmap
        # Find the combination with the highest stroke percentage
        if not heatmap_data.empty:
            max_idx = heatmap_data['stroke_percent'].idxmax()
            max_row = heatmap_data.loc[max_idx]
            top_bmi_cat = max_row['bmi_category']
            top_glucose_cat = max_row['glucose_category']
            st.caption(f"The highest risk combination is {top_bmi_cat} BMI with {top_glucose_cat} glucose levels")

        
        # Chart 11: Stroke Rate by Medical Risk Factor
        # Compares major health risk factors to see which drives stroke risk most
        with st.container():
            st.subheader("Stroke Rate by Medical Risk Factor")
            risk_rates = {
                'Age > 60': filtered_df[filtered_df['age'] > 60]['stroke'].mean(),
                'Hypertension': filtered_df[filtered_df['hypertension'] == 1]['stroke'].mean(),
                'Heart Disease': filtered_df[filtered_df['heart_disease'] == 1]['stroke'].mean(),
                'Glucose > 140': filtered_df[filtered_df['avg_glucose_level'] > 140]['stroke'].mean(),
                'BMI > 30': filtered_df[filtered_df['bmi'] > 30]['stroke'].mean()
            }
            # Convert rates to percentages and round
            for key in risk_rates:
                risk_rates[key] = round(risk_rates[key] * 100, 2)
            risk_df = pd.DataFrame(list(risk_rates.items()), columns=['Risk Factor', 'Stroke Rate (%)'])
            risk_df = risk_df.sort_values(by='Stroke Rate (%)', ascending=False)
            # Use the existing color palette for discrete colours
            bar_colors = color_palette[:len(risk_df)]
            color_mapping = {factor: bar_colors[i % len(bar_colors)] for i, factor in enumerate(risk_df['Risk Factor'])}
            fig_risk = px.bar(
                risk_df,
                x='Risk Factor',
                y='Stroke Rate (%)',
                text='Stroke Rate (%)',
                labels={'Stroke Rate (%)': 'Stroke Rate (%)', 'Risk Factor': 'Medical Risk Factor'},
                color='Risk Factor',
                color_discrete_map=color_mapping
            )
            fig_risk.update_traces(textposition='outside')
            fig_risk.update_layout(
                template='plotly_dark',
                showlegend=False,
                yaxis=dict(range=[0, risk_df['Stroke Rate (%)'].max() + 5])
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            top_factor = risk_df.iloc[0]['Risk Factor']
            st.caption(f"{top_factor} has the highest stroke rate among these medical risk factors")
