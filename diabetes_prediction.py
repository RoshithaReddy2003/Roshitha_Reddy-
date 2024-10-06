# diabetes_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Split dataset into features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)

# Save the model
joblib.dump(model, 'diabetes_model.pkl')
