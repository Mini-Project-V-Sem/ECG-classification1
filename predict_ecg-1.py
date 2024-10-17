import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from pymongo import MongoClient
from urllib.parse import quote_plus
import bcrypt
import certifi
import ssl

# Connect to MongoDB
client = MongoClient("mongodb+srv://Mandar_Wagh:mandar%401107@ecg-users.i4kje.mongodb.net/?retryWrites=true&w=majority")
db = client.ecg_users
users_collection = db.users

# Define helper functions for user registration and authentication
def register_user(username, password):
    if users_collection.find_one({"username": username}):
        return False  # Username already exists
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({"username": username, "password": hashed_password})
    return True

def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return True
    return False


# Load your RNN model
RNN_model = load_model('RNN_model_updated.keras')

# User login/registration section
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login_page():
    if not st.session_state['logged_in']:
        st.title("ECG Classification with RNN Model")

        choice = st.selectbox('Login or Register', ['Login', 'Register'])

        if choice == 'Register':
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Register"):
                if register_user(username, password):
                    st.success("Account created!")
                else:
                    st.error("Username already exists!")

        elif choice == 'Login':
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if authenticate_user(username, password):
                    st.success("Login successful!")
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    ecg_classification_page()  # Redirect to the ECG classification page after login
                else:
                    st.error("Incorrect credentials")
    else:
        ecg_classification_page()

def ecg_classification_page():
    st.subheader(f"Welcome, {st.session_state['username']}!")
    
    # Input Section
    st.write("Input your ECG data for classification:")

    # Allow the user to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file containing ECG data (187 values expected)", type="csv")

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file without a header
            data = pd.read_csv(uploaded_file, header=None)

            # Flatten data to a 1D array
            if data.shape[1] == 187:
                st.success("File successfully uploaded and contains 187 values in a single row.")
                input_data = data.values.flatten()

            elif data.shape[0] == 187 and data.shape[1] == 1:
                st.success("File successfully uploaded and contains 187 values in a single column.")
                input_data = data.values.flatten()

            elif data.shape[0] == 1 and data.shape[1] == 1:
                # Single cell containing comma-separated values
                input_data = data.iloc[0, 0].split(',')
                input_data = list(map(float, input_data))

            else:
                st.error(f"Uploaded file has {data.shape[1]} columns and {data.shape[0]} rows. Only data with exactly 187 values is acceptable.")
                input_data = None

            if input_data is not None:
                if len(input_data) > 187:
                    st.warning(f"Uploaded file contains {len(input_data)} values. Trimming to 187 values.")
                    input_data = input_data[:187]
                elif len(input_data) < 187:
                    st.error(f"Uploaded file contains {len(input_data)} values, but exactly 187 values are required. Please provide a complete dataset.")
                    input_data = None

                if input_data is not None:
                    # Display the first few values for verification
                    st.write(f"First 5 values: {input_data[:5]}")

                    # Preprocess the input data
                    try:
                        # Convert the input data to float
                        input_data = np.array(input_data, dtype=float)

                        # Normalize input data
                        input_data_normalized = input_data / np.max(np.abs(input_data))
                        input_data_reshaped = input_data_normalized.reshape(1, 187, 1)  # RNN expects (batch_size, time_steps, features)

                        # Predict with the RNN model
                        rnn_probs = RNN_model.predict(input_data_reshaped)
                        predicted_class = np.argmax(rnn_probs, axis=1)[0]

                        st.success(f"Predicted class: {predicted_class}")

                    except ValueError:
                        st.error("Error processing the numeric values. Ensure the data is in the correct format.")
            else:
                st.error("No valid ECG data found in the uploaded file.")
        except Exception as e:
            st.error(f"Error reading the file: {e}")

    else:
        st.write("Please upload a CSV file with exactly 187 comma-separated ECG values.")

# Run the login page
login_page()
