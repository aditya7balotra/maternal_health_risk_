from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset and preprocess
df = pd.read_csv('maternal_health_risk_dataset.csv')
encoder = LabelEncoder()
df['RiskLevel'] = df['RiskLevel'].apply(lambda x: 0 if x == 'low risk' else 1 if x == 'mid risk' else 2)
x = df.drop('RiskLevel', axis=1)
y = df['RiskLevel']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train the Decision Tree model
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    age = int(request.form['age'])
    sysbp = int(request.form['systolic_bp'])
    disbp = int(request.form['diastolic_bp'])
    bs = int(request.form['bs'])
    bdytem = int(request.form['body_temp'])
    hrt = int(request.form['heart_rate'])

    # Make prediction using the trained Decision Tree model
    query = np.array([[age, sysbp, disbp, bs, bdytem, hrt]])
    prediction = tree.predict(query)
    
    # Map predicted class labels to risk levels
    risk_levels = {0: 'low risk', 1: 'mid risk', 2: 'high risk'}
    predicted_risk = risk_levels[prediction[0]]

    # Render the prediction result
    return render_template('result.html', prediction=predicted_risk)

if __name__ == '__main__':
    app.run(debug=True)
