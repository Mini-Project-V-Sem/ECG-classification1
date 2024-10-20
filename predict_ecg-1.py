import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from pymongo import MongoClient
import bcrypt
import time
from datetime import datetime

# MongoDB connection
client = MongoClient("mongodb+srv://Mandar_Wagh:mandar%401107@ecg-users.i4kje.mongodb.net/?retryWrites=true&w=majority")
db = client.ecg_users
users_collection = db.users

# Helper functions
def register_user(username, password, name, dob, gender, contact_info, consent):
    if users_collection.find_one({"username": username}):
        return False  # Username already exists
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Convert 'dob' to string or datetime
    dob_str = dob.strftime("%Y-%m-%d")  # Converts to 'YYYY-MM-DD' format
    
    users_collection.insert_one({
        "username": username,
        "password": hashed_password,
        "name": name,
        "dob": dob_str,
        "gender": gender,
        "contact_info": contact_info,
        "consent": consent
    })
    return True

def authenticate_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return True
    return False

# Load your RNN model
RNN_model = load_model('RNN_model_updated.keras')

# Custom CSS for buttons and title
button_style = """
    <style>
        .btn {
            padding: 10px 20px;
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
            transition: 0.3s;
        }
        .btn:hover {
            box-shadow: 4px 4px 8px rgba(0, 0, 0, 0.5);
        }
        .title {
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
"""

st.markdown(button_style, unsafe_allow_html=True)

# User login/registration section
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login_page():
    if not st.session_state['logged_in']:
        st.markdown("<h1 class='title'>ECG Classifiers</h1>", unsafe_allow_html=True)

        choice = st.selectbox('Login or Register', ['Login', 'Register'])

        if choice == 'Register':
            # Registration fields
            name = st.text_input("Name")
            dob = st.date_input("Date of Birth")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            contact_info = st.text_input("Contact Info")
            consent = st.checkbox("I consent to the use of my information for research and analysis")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Register", key="register"):
                if not consent:
                    st.error("You must consent to the use of your information to proceed.")
                elif register_user(username, password, name, dob, gender, contact_info, consent):
                    st.success("Account created! Please log in.")
                else:
                    st.error("Username already exists!")

        elif choice == 'Login':
            # Login fields
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login", key="login"):
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
            st.write("File successfully uploaded. Processing...")

            data = pd.read_csv(uploaded_file, header=None)

            st.write(f"File shape: {data.shape}")

            input_data = None

            if data.shape[1] == 187:
                st.success("File successfully uploaded and contains 187 values in a single row.")
                input_data = data.values.flatten()
            elif data.shape[1] == 188:
                st.warning("File contains 188 values, trimming the last value to fit the expected 187 values.")
                input_data = data.values.flatten()[:187]
            elif data.shape[0] == 187 and data.shape[1] == 1:
                st.success("File successfully uploaded and contains 187 values in a single column.")
                input_data = data.values.flatten()
            elif data.shape[0] == 1 and data.shape[1] == 1:
                input_data = data.iloc[0, 0].split(',')
                input_data = list(map(float, input_data))

            if input_data is not None:
                if len(input_data) > 187:
                    st.warning(f"Uploaded file contains {len(input_data)} values. Trimming to 187 values.")
                    input_data = input_data[:187]
                elif len(input_data) < 187:
                    st.error(f"Uploaded file contains {len(input_data)} values, but exactly 187 values are required.")
                    input_data = None

                if input_data is not None:
                    st.write(f"First 5 values: {input_data[:5]}")

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for percent_complete in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(percent_complete + 1)
                        status_text.text(f"Processing... {percent_complete + 1}% complete")

                    input_data = np.array(input_data, dtype=float)
                    input_data_normalized = input_data / np.max(np.abs(input_data))
                    input_data_reshaped = input_data_normalized.reshape(1, 187, 1)

                    # Predict class using the RNN model
                    rnn_probs = RNN_model.predict(input_data_reshaped)
                    predicted_class = np.argmax(rnn_probs, axis=1)[0]

                    st.success(f"Predicted class: {predicted_class}")

                    # Save predicted class to MongoDB (convert to int to avoid numpy type issue)
                    username = st.session_state['username']
                    users_collection.update_one(
                        {"username": username}, 
                        {"$set": {"predicted_class": int(predicted_class)}}
                    )

                    # Displaying classes and descriptions
                    button_style = """
                        <style>
                            .class-button {
                                padding: 10px;
                                width: 120px;
                                height: 50px;
                                border-radius: 10px;
                                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
                                margin: 5px;
                                text-align: center;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            }
                            .highlighted {
                                background-color: #E8F9EE;
                            }
                            .regular {
                                background-color: #FFFFFF;
                            }
                        </style>
                    """

                    st.markdown(button_style, unsafe_allow_html=True)

                    st.write("Classification Results:")

                    classes = [
                        "Class 0: Normal", 
                        "Class 1: Supraventricular", 
                        "Class 2: Ventricular", 
                        "Class 3: Fusion",
                        "Class 4: Unknown"
                    ]
                    descriptions = [
                        "Normal sinus rhythm: The heart beats in a normal rhythm without arrhythmias.",
                        "Supraventricular Arrhythmia: Abnormal fast rhythms originating above the heart’s ventricles.",
                        "Ventricular Arrhythmia: Irregular heartbeats that start in the lower chambers of the heart (ventricles).",
                        "Fusion Beat: A fusion of two heartbeats, one from normal rhythm and one from an ectopic source.",
                        "Unclassifiable Beats: Beats that cannot be classified into the known categories."
                    ]

                    cols = st.columns(5)
                    for i, col in enumerate(cols):
                        if i == predicted_class:
                            col.markdown(
                                f"<div class='class-button highlighted' title='{descriptions[i]}'>{classes[i]}</div>", 
                                unsafe_allow_html=True
                            )
                        else:
                            col.markdown(
                                f"<div class='class-button regular' title='{descriptions[i]}'>{classes[i]}</div>", 
                                unsafe_allow_html=True
                            )

                    st.markdown("<br><br>", unsafe_allow_html=True)

                    st.markdown(f"""
                        <div style="border: 2px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
                        <strong>{classes[predicted_class]}</strong>: {descriptions[predicted_class]}
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# Run the login page
login_page()
