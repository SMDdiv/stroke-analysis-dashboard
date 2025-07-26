import streamlit as st
import pandas as pd
import plotly.express as px

# Sidebar Styling
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
                   page_icon="📊")

st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to", ["Description", "Model", "Dashboard"])

@st.cache_data
def load_data(path: str = "healthcare-dataset-stroke-data.csv") -> pd.DataFrame | None:
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
    st.title("📄 Description Page")
    st.write("📝 Write your project description here.")

elif page == "Model":
    st.title("Model Page")
    st.write("Add your model details here.")

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

        # Chart 2: Stroke Distribution by Gender
        # Shows male vs female stroke rate using a donut-style pie chart
        with st.container():
            st.subheader("Stroke Rate by Glucose Level")
            sorted_glucose = glucose_stroke.sort_values('stroke_percent', ascending=False)
            fig_glucose = px.bar(
                sorted_glucose,
                y='glucose_category',
                x='stroke_percent',
                orientation='h',
                color='glucose_category',
                labels={'stroke_percent': 'Stroke Rate (%)', 'glucose_category': 'Glucose Level'},
                text=sorted_glucose['stroke_percent'].round(1),
                color_discrete_sequence=color_palette
            )
            fig_glucose.update_layout(template='plotly_dark', showlegend=False)
            fig_glucose.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
            st.plotly_chart(fig_glucose, use_container_width=True)

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

        st.subheader("Stroke Rate by BMI and Glucose Level")
        heatmap_data = filtered_df.groupby(['bmi_category', 'glucose_category'], observed=False)['stroke'].mean().reset_index()
        heatmap_data['stroke_percent'] = heatmap_data['stroke'] * 100
        heatmap_pivot = heatmap_data.pivot(index='glucose_category', columns='bmi_category', values='stroke_percent')
        fig_heatmap = px.imshow(heatmap_pivot, text_auto=True, color_continuous_scale='Teal')
        fig_heatmap.update_layout(template='plotly_dark', title='Stroke Rate (%) by BMI and Glucose Category')
        st.plotly_chart(fig_heatmap, use_container_width=True)

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
