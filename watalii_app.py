import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import os 

# Load the data
data = pd.read_csv('V:\\my project\\app\\Clean_Kenya_Tourism_datasets (2).csv')

# Data Preprocessing
data['travel_with'] = data['travel_with'].replace(np.nan, 'Alone')
data['total_female'] = data['total_female'].replace(np.nan, 1.0)
data['total_male'] = data['total_male'].replace(np.nan, 1.0)
data['most_impressing'] = data['most_impressing'].replace(np.nan, 'No comments')
data['age_group'] = data['age_group'].replace('24-Jan', '1-24')

# Convert float columns to integer
data["total_female"] = data['total_female'].astype('int')
data["total_male"] = data['total_male'].astype('int')
data["nights_spent"] = data['nights_spent'].astype('int')

# Generate new features
data["total_people"] = data["total_female"] + data["total_male"]
data["total_nights"] = data["nights_spent"]

# Encode categorical features
for colname in data.select_dtypes("object"):
    data[colname], _ = data[colname].factorize()

# Model Building
x = data.drop(['total_cost'], axis=1)
y = data['total_cost']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Train the XGBoost model
model = XGBRegressor()
model.fit(X=x_train, y=y_train)

# Create a Streamlit app
st.title('Kenya Tourism Prediction App')

# Add a sidebar for user input
st.sidebar.header('User Input Features')

# Function to get user input
def get_user_input():
    total_female = st.sidebar.slider('Total Female', int(data['total_female'].min()), int(data['total_female'].max()), int(data['total_female'].mean()))
    total_male = st.sidebar.slider('Total Male', int(data['total_male'].min()), int(data['total_male'].max()), int(data['total_male'].mean()))
    nights_spent = st.sidebar.slider('Nights Spent', int(data['nights_spent'].min()), int(data['nights_spent'].max()), int(data['nights_spent'].mean()))
    travel_with = st.sidebar.selectbox('Travel With', data['travel_with'].unique())
    most_impressing = st.sidebar.selectbox('Most Impressing', data['most_impressing'].unique())
    age_group = st.sidebar.selectbox('Age Group', data['age_group'].unique())
    features = {'total_female': total_female,
                'total_male': total_male,
                'nights_spent': nights_spent,
                'travel_with': travel_with,
                'most_impressing': most_impressing,
                'age_group': age_group}
    return pd.DataFrame(features, index=[0])

# Get user input
user_input = get_user_input()

# Perform prediction
prediction = model.predict(user_input)

# Display results
st.header("Results")
st.write("You are expected to spend: {}".format(prediction))

