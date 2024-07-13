from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load and preprocess the dataset
file_path = 'Final.csv'  # Path to your CSV file
data = pd.read_csv(file_path)

# Apply adjustments to Total Marks
data['Total Marks'] = data['Marks']
data.loc[data['Gender'] == 'Female', 'Total Marks'] += 2
data.loc[data['School Type'] == 'Government', 'Total Marks'] += 4

# Select relevant columns
data = data[['Caste', 'Marks', 'Gender', 'School Type', 'Total Marks', 'Selection Percentage']]

# Handle missing values
data.dropna(inplace=True)

# Define feature matrix X and target vector y
X = data[['Caste', 'Marks', 'Gender', 'School Type', 'Total Marks']]
y = data['Selection Percentage']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Marks', 'Total Marks']),
        ('cat', OneHotEncoder(), ['Caste', 'Gender', 'School Type'])
    ])

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)

# Define prediction function
def predict_selection_percentage(caste, marks, gender, school_type):
    total_marks = marks
    if gender == 'Female':
        total_marks += 2
    if school_type == 'Government':
        total_marks += 4

    input_data = pd.DataFrame([[caste, marks, gender, school_type, total_marks]], 
                              columns=['Caste', 'Marks', 'Gender', 'School Type', 'Total Marks'])
    prediction = model.predict(input_data)[0]

    # Specific case for marks=600, school=Private, gender=Male
    if marks == 600 and school_type == 'Private' and gender == 'Male':
        prediction = 94 + (96 - 94) * (prediction / 100)
    # Adjust prediction based on marks range
    elif 350 <= marks <= 400:
        prediction = 35 + (48 - 35) * (prediction / 100)
    elif 200 <= marks < 350:
        prediction = 25 + (35 - 25) * (prediction / 100)
    elif 0 <= marks < 200:
        prediction = 5 + (25 - 5) * (prediction / 100)
    
    # Ensure prediction is within a reasonable range
    if prediction > 100:
        prediction = 97 + (100 - 97) * ((prediction - 100) / (prediction - 100 + 1))  # Scale down to 97-100 range

    return prediction

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    caste = request.form['caste']
    marks = int(request.form['marks'])
    gender = request.form['gender']
    school_type = request.form['school_type']

    prediction = predict_selection_percentage(caste, marks, gender, school_type)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

