import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from pymongo import MongoClient
import bcrypt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import time
import io
import base64
import plotly.graph_objects as go
import tempfile
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer,Table, Image,TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

# Set page config
st.set_page_config(page_title="ECG Heartbeat Classifier", page_icon="❤️", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://lottie.host/a8adf41c-15c6-462a-9462-0776c9ea7522/0u2xKRXTS4.json"
lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)


st_lottie(lottie_hello, key="hello")


# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .css-1v3fvcr {
        background-color: #f9f9f9;
    }
    .title {
        font-size: 3em;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
    }
    .subtitle {
        font-size: 1.5em;
        color: #34495e;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# MongoDB connection (replace with your actual connection string)
client = MongoClient("mongodb+srv://Mandar_Wagh:mandar%401107@ecg-users.i4kje.mongodb.net/?retryWrites=true&w=majority")
db = client.ecg_users
users_collection = db.users

# Load RNN model (ensure the model file is in the same directory)
RNN_model = load_model('RNN_model_updated.keras')

# Helper functions
def register_user(username, password, name, dob, gender, contact_info, consent):
    if users_collection.find_one({"username": username}):
        return False
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        "username": username,
        "password": hashed_password,
        "name": name,
        "dob": dob.strftime("%Y-%m-%d"),
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


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_radar_chart(prediction):
    # Create radar chart for class probabilities
    classes = ["Normal", "Premature", "Supraventricular", "Fusion", "Unclassified"]
    angles = np.linspace(0, 2*np.pi, len(classes), endpoint=False)
    
    # Close the plot by appending first value
    values = list(prediction[0] * 100)
    values += values[:1]
    angles = list(angles)
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes)
    ax.set_title("Classification Confidence Distribution")
    
    # Save to buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def generate_pdf_report(predicted_class, classes, descriptions, input_data, confidence, prediction=None, username=""):
    try:
        # Create the PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        ))
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12
        ))
        styles.add(ParagraphStyle(
            name='Body',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12
        ))

        # Build the document
        story = []

        # Title
        story.append(Paragraph("ECG Heartbeat Classification Report", styles['CustomTitle']))

        # Patient Information
        story.append(Paragraph("Patient Information", styles['SectionHeader']))
        patient_data = [
            ["Patient Name:", username if username else "Anonymous"],
            ["Date:", datetime.now().strftime("%Y-%m-%d")],
            ["Time:", datetime.now().strftime("%H:%M:%S")]
        ]
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))

        # Classification Results
        story.append(Paragraph("Classification Results", styles['SectionHeader']))
        results_data = [
            ["Predicted Class:", classes[predicted_class]],
            ["Confidence:", f"{confidence:.2f}%"],
            ["Description:", descriptions[predicted_class]]
        ]
        results_table = Table(results_data, colWidths=[2*inch, 4*inch])
        results_table.setStyle(TableStyle([
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1,-1), 12),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 20))

        # ECG Signal Plot
        story.append(Paragraph("ECG Signal Analysis", styles['SectionHeader']))
        plt.figure(figsize=(8, 4))
        # Remove .hexval() and pass the hex color directly
        plt.plot(input_data, color='#2980b9')
        plt.title('ECG Signal')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot to buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        # Add ECG plot to PDF
        img = Image(img_buffer, width=6.5*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 20))



        # Add radar chart if prediction data is available
        if prediction is not None:
            story.append(Paragraph("Classification Confidence Distribution", styles['SectionHeader']))
            radar_buffer = create_radar_chart(prediction)
            radar_img = Image(radar_buffer, width=4*inch, height=4*inch)
            story.append(radar_img)
            story.append(Spacer(1, 20))

        # Recommendations
        story.append(Paragraph("Recommendations", styles['SectionHeader']))
        recommendations = {
            0: " Continue with a healthy lifestyle and regular check-ups to maintain cardiovascular health.",
            1: "If the premature beats are infrequent and not accompanied by any symptoms, they are usually harmless. However, if they are persistent, accompanied by chest pain, dizziness, or shortness of breath, seek medical consultation for further evaluation.",
            2: "If supraventricular arrhythmias are detected, it is important to undergo further tests, including an echocardiogram or Holter monitor, to assess the severity and determine the appropriate treatment. If left untreated, some forms of supraventricular arrhythmias can lead to more serious complications, including stroke.",
            3: "Fusion beats can sometimes be benign, especially if they occur sporadically, but they can also signal more significant electrical conduction problems. Further evaluation through an ECG and potentially an electrophysiological study (EPS) is recommended to determine if there is an underlying conduction defect that requires treatment.",
            4: "Since the exact nature of the unclassified beat is unclear, it is important to follow up with further diagnostic tests such as a 24-hour Holter monitor, stress test, or a more detailed electrophysiology study. This will help determine if the abnormality is transient, benign, or indicative of a more serious condition that requires medical attention."
        }
        story.append(Paragraph(recommendations[predicted_class], styles['Body']))
        
        # Disclaimer
        story.append(Spacer(1, 20))
        story.append(Paragraph("Disclaimer", styles['SectionHeader']))
        disclaimer_text = ("This report is generated by an AI-based ECG analysis system and should be used for "
                         "informational purposes only. It is not intended to be a substitute for professional medical "
                         "advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified "
                         "health provider with any questions you may have regarding a medical condition.")
        story.append(Paragraph(disclaimer_text, styles['Body']))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        raise
# Streamlit app
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    st.markdown("<h1 class='title'>ECG Heartbeat Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Advanced AI-powered ECG Analysis</p>", unsafe_allow_html=True)

    if not st.session_state['logged_in']:
        login_page()
    else:
        app_page()

def login_page():
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            if authenticate_user(username, password):
                st.success("Login successful!")
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                time.sleep(1)
            else:
                st.error("Invalid username or password.")
    
    with tab2:
        st.subheader("Register")
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            name = st.text_input("Full Name")
            dob = st.date_input("Date of Birth")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            contact_info = st.text_input("Contact Information")
            consent = st.checkbox("I consent to the use of my data for research purposes")
            submitted = st.form_submit_button("Register")
            if submitted:
                if register_user(new_username, new_password, name, dob, gender, contact_info, consent):
                    st.success("Registration successful! You can now log in.")
                else:
                    st.error("Registration failed. Username may already exist.")

def app_page():
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    page = st.sidebar.radio("Navigation", ["ECG Classification", "User Profile", "About"])
    
    if page == "ECG Classification":
        classification_page()
    elif page == "User Profile":
        user_profile_page()
    elif page == "About":
        about_page()
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.success("Logged out successfully!")
        time.sleep(1)
        st.experimental_rerun()

def classification_page():
    st.header("ECG Heartbeat Classification")
    
    uploaded_file = st.file_uploader("Upload a CSV file containing ECG data (1 row only)", type="csv")
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file, header=None)
            
            if data.shape[0] != 1:
                st.error("The uploaded file must contain only one row of data.")
                return

            input_string = data.iloc[0, 0]
            input_data = np.array([float(x) for x in input_string.split(',')])

            desired_length = 100
            if len(input_data) < desired_length:
                input_data = np.pad(input_data, (0, desired_length - len(input_data)), mode='constant')
            elif len(input_data) > desired_length:
                input_data = input_data[:desired_length]

            input_data_reshaped = input_data.reshape((1, -1, 1))
            prediction = RNN_model.predict(input_data_reshaped)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            classes = [
                "Normal Beat",
                "Premature Beat",
                "Supraventricular Beat",
                "Fusion Beat",
                "Unclassified Beat"
            ]
            descriptions = [
                "This indicates that the heart's electrical activity is functioning as\n expected, with the heart rhythm following the regular sequence \nof atrial and ventricular contraction. There are no signs \nof arrhythmia, and the heart rate and rhythm are\n consistent with a healthy individual. If you are experiencing no symptoms or discomfort, there is no cause for concern.",
                "Premature heartbeats, also known as premature atrial contractions (PACs) \nor premature ventricular contractions (PVCs), are early beats \nthat disrupt the normal rhythm of the heart. \nWhile they are often benign and may be caused by stress, \ncaffeine, or fatigue, persistent or frequent premature \nbeats may signal an underlying issue, such as \nelectrolyte imbalance, heart disease, or other arrhythmic \ndisorders.",
                "Supraventricular beats originate above the ventricles, typically in \nthe atria or the AV node. This can include arrhythmias like atrial \nfibrillation, atrial flutter, or paroxysmal supraventricular tachycardia \n(PSVT). These types of arrhythmias can cause the\n heart to beat too fast or irregularly, which may result in symptoms \nsuch as palpitations, shortness of breath, dizziness, or fainting.",
                "A fusion beat occurs when the heart receives two electrical impulses \nat nearly the same time—one from the normal conduction pathway and \nthe other from an ectopic focus (an abnormal\n area of the heart generating an impulse). This results in a \nhybrid contraction of the heart muscle. Fusion beats may be a sign of an underlying conduction disorder, such as bundle branch block or heart block.",
                " An unclassified beat refers to a rhythm or anomaly that does not\n fall into a specific category recognized by the ECG analysis system. \nThese beats could be caused by a variety of \nfactors, including unusual heart rhythms, poor quality of the\n ECG signal, or a new, unexplored type of arrhythmia that doesn't fit the common diagnostic criteria."
            ]
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ECG Signal Visualization")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=input_data, mode='lines', name='ECG Signal'))
                fig.update_layout(title='ECG Signal', xaxis_title='Sample', yaxis_title='Amplitude')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Classification Results")
                st.markdown(f"**Predicted Class:** {classes[predicted_class]}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                st.markdown(f"**Description:** {descriptions[predicted_class]}")

                # Radar chart for class probabilities
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=prediction[0] * 100,
                    theta=classes,
                    fill='toself',
                    name='Class Probabilities'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            if st.button("Generate PDF Report"):
                try:
                    # Generate the PDF buffer
                    pdf_buffer = generate_pdf_report(predicted_class, classes, descriptions, input_data, confidence)
                    
                    # Pass the bytes of the PDF buffer directly to the download button
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer.getvalue(),  # Use .getvalue() to extract the content of the buffer
                        file_name="ecg_report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"An error occurred while generating the PDF: {str(e)}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
def user_profile_page():
    st.header("User Profile")
    user = users_collection.find_one({"username": st.session_state['username']})
    if user:
        st.write(f"**Name:** {user['name']}")
        st.write(f"**Date of Birth:** {user['dob']}")
        st.write(f"**Gender:** {user['gender']}")
        st.write(f"**Contact Information:** {user['contact_info']}")
        
        if st.button("Delete Account"):
            if st.checkbox("I understand this action is irreversible"):
                users_collection.delete_one({"username": st.session_state['username']})
                st.session_state['logged_in'] = False
                st.session_state['username'] = None
                st.success("Your account has been deleted. You will be logged out.")
                time.sleep(2)
    else:
        st.error("User profile not found. Please contact support.")

def about_page():
    st.header("About ECG Heartbeat Classifier")
    st.write("""
    The ECG Heartbeat Classifier is an advanced AI-powered tool designed to analyze and classify electrocardiogram (ECG) signals. 
    Our system uses state-of-the-art machine learning techniques to identify various types of heartbeats, including:

    - Normal Beats
    - Premature Beats
    - Supraventricular Beats
    - Fusion Beats
    - Unclassified Beats

    This tool is intended for research and educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
    """)
    
    st.subheader("How It Works")
    st.write("""
    1. Upload your ECG data in CSV format.
    2. Our AI model analyzes the signal patterns.
    3. The system classifies the heartbeat and provides a confidence score.
    4. You can view the results and generate a detailed PDF report.
    """)
    
    st.subheader("Privacy and Data Usage")
    st.write("""
    We take your privacy seriously. All uploaded ECG data is processed securely and is not stored on our servers after analysis. 
    User account information is encrypted and stored securely. By using this service, you agree to our terms of service and privacy policy.
    """)

if __name__ == "__main__":
    main()
