import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from pymongo import MongoClient
import bcrypt
import time
client = MongoClient("mongodb+srv://Mandar_Wagh:mandar%401107@ecg-users.i4kje.mongodb.net/?retryWrites=true&w=majority")
db = client.ecg_users
users_collection = db.users

# Helper functions
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
        st.title("ECG Classifiers")

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
                    st.write(f"First 5 values: {input_data[:5]}")

                    # Simulate a loading process with fixed speed of 5 seconds (0.05 seconds per step)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for percent_complete in range(100):
                        time.sleep(0.05)  # 5 seconds total (0.05 * 100)
                        progress_bar.progress(percent_complete + 1)
                        status_text.text(f"Processing... {percent_complete + 1}% complete")

                    # Normalize input data
                    input_data = np.array(input_data, dtype=float)
                    input_data_normalized = input_data / np.max(np.abs(input_data))
                    input_data_reshaped = input_data_normalized.reshape(1, 187, 1)

                    # Predict with the RNN model
                    rnn_probs = RNN_model.predict(input_data_reshaped)
                    predicted_class = np.argmax(rnn_probs, axis=1)[0]

                    # Show predicted class only after progress is fully completed
                    st.success(f"Predicted class: {predicted_class}")

                    # Styling for the buttons (all classes)
                    button_style = """
                        <style>
                            .class-button {
                                padding: 10px;
                                width: 120px;  /* Fixed width */
                                height: 50px;  /* Fixed height */
                                border-radius: 10px;
                                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
                                margin: 5px;
                                text-align: center;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            }
                            .highlighted {
                                background-color: #E8F9EE; /* Color for the predicted class */
                            }
                            .regular {
                                background-color: #FFFFFF; /* Default color for other classes */
                            }
                        </style>
                    """
                    st.markdown(button_style, unsafe_allow_html=True)

                    # Display 5 classes and highlight the predicted one
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
                        "Unclassifiable Beats: Beats that cannot be classified into the known categories, possibly due to noise or unknown abnormalities."
                    ]

                    # Displaying the classes with styling
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

                    # Add space between the buttons and the subheader
                    st.markdown("<br><br>", unsafe_allow_html=True)  # Adds gap

                    # Display the class number and description inside a styled box
                    st.markdown(f"""
                        <div style="border: 2px solid #E8F9EE; border-radius: 10px; padding: 10px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); background-color: #E8F9EE;">
                            <h5 style="margin-bottom: 5px;">Predicted Class {predicted_class} :</h5>
                            <p>{descriptions[predicted_class]}</p>
                        </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error reading the file: {e}")

# Run the login page
login_page()
