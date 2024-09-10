import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Assuming the model is saved in the current directory
# If not, adjust the path accordingly
model_filename = 'logistic_model.pkl'

try:
    # Load the trained model
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    st.write("Model loaded successfully!")

    # Now you can use the loaded_model within your Streamlit app
    # For example:
    
    st.write("Model Coefficients:")
    coefficients = pd.DataFrame({"Feature": loaded_model.feature_names_in_, "Coefficient": loaded_model.coef_[0]})
    st.write(coefficients.sort_values(by='Coefficient', ascending=False))

except FileNotFoundError:
    st.write(f"Error: Model file '{model_filename}' not found.")
    # Handle the case where the model file is not found
    # For example, you could display a message to the user or use a default model


# Title and Inputs
st.title("Logistic Regression Predictor")

# Create input fields for user to input data
input_1 = st.number_input("Input feature 1", value=0)
input_2 = st.number_input("Input feature 2", value=0)

# Predict button
if st.button("Predict"):
    features = np.array([[input_1, input_2]])  # Ensure inputs match model training features
    prediction = model.predict(features)
    st.write("Prediction:", "Survived" if prediction[0] == 1 else "Not Survived")
