import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('Crop_recommendation.csv')

X = data.drop(columns=['label'])
y = data['label']

X = pd.get_dummies(X)

model = DecisionTreeClassifier()
model.fit(X, y)

with open('crop_recommendation_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
