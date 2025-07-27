# stroke_model.py

import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

# ========== Load dataset ==========
df = pd.read_csv(r'healthcare-dataset-stroke-data.csv')

# ========== Feature Engineering ==========
# Create a binary feature for smoking status
# 1 if smokes or formerly smoked, 0 otherwise
# Remove rows with missing BMI values
# Remove rows where smoking status is 'Unknown'
df['is_smoker'] = df['smoking_status'].apply(lambda x: 1 if x in ['smokes', 'formerly smoked'] else 0)
df = df[df['bmi'].notna()]
df = df[df['smoking_status'] != 'Unknown']

# Function to classify life stage based on age
def life_stage_classification(age):
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

# Apply life stage classification
df['life_stage'] = df['age'].apply(life_stage_classification)

# Function to categorize BMI
def categorize_bmi(bmi):
    if pd.isnull(bmi):
        return 'Unknown'
    elif bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

# Apply BMI categorization
df['bmi_category'] = df['bmi'].apply(categorize_bmi)

# Function to categorize glucose level
def glucose_category(val):
    if val < 70:
        return 'Hypoglycemia'
    elif val <= 99:
        return 'Normal'
    elif val <= 125:
        return 'Prediabetes'
    else:
        return 'Diabetes'

# Apply glucose level categorization
df['glucose_level_category'] = df['avg_glucose_level'].apply(glucose_category)

# ========== Undersample to balance stroke data ==========
# Balance the dataset by undersampling the majority class (stroke=0)
stroke_0 = df[df['stroke'] == 0]
stroke_1 = df[df['stroke'] == 1]
stroke_0_sampled = stroke_0.sample(n=len(stroke_1), random_state=42)
df_balanced = pd.concat([stroke_0_sampled, stroke_1])

# ========== Define features & target ==========
# Select features and target variable for modeling
features = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'is_smoker',
    'gender', 'ever_married', 'work_type', 'Residence_type',
    'life_stage', 'bmi_category', 'glucose_level_category'
]
target = 'stroke'

X = df_balanced[features]
y = df_balanced[target]

# ========== Split data ==========
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Preprocessing Pipelines ==========
# Define preprocessing for numeric features (impute missing values with mean)
numeric_features = ['age', 'avg_glucose_level', 'bmi']
numeric_transformer = SimpleImputer(strategy='mean')

# Define preprocessing for categorical features (impute missing values and one-hot encode)
categorical_features = [
    'hypertension', 'heart_disease', 'is_smoker',
    'gender', 'ever_married', 'work_type', 'Residence_type',
    'life_stage', 'bmi_category', 'glucose_level_category'
]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ========== Build and Train Model ==========
# Create a pipeline with preprocessing and RandomForest classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Save test set and predictions for Streamlit visualizations
X_test.to_csv('X_test.csv', index=False)
pd.DataFrame({'y_test': y_test}).to_csv('y_test.csv', index=False)
pd.DataFrame({'y_pred': y_pred, 'y_proba': y_proba}).to_csv('y_pred_proba.csv', index=False)

# Print classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# ========== Feature Importance ==========
# Get feature names after one-hot encoding
onehot_feature_names = model.named_steps['preprocessor'].transformers_[1][1] \
    .named_steps['onehot'].get_feature_names_out(categorical_features)
all_features = np.concatenate([numeric_features, onehot_feature_names])
importances = model.named_steps['classifier'].feature_importances_

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# ========== Plot Top Features ==========
# Plot the top 15 important features using Plotly
fig = px.bar(
    importance_df.head(15),
    x='Importance',
    y='Feature',
    orientation='h',
    title='Top 15 Important Features for Stroke Prediction',
    labels={'Importance': 'Importance', 'Feature': 'Feature'}
)
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()

# Save the trained model to a file
joblib.dump(model, 'model.pkl')
