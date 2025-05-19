from flask import Flask, render_template, request, redirect, send_file, url_for, flash, session, send_from_directory
from pymongo import MongoClient
from werkzeug.security import check_password_hash, generate_password_hash
from bson.objectid import ObjectId
import os
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import torch
from openai import OpenAI
from flask import jsonify
from datetime import datetime
from reportlab.platypus import Paragraph, SimpleDocTemplate, HRFlowable, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from werkzeug.utils import secure_filename

csv_file = r'D:\testing-final\testing-final\all_symptoms.csv'  # Replace with your actual file path
# Set the folder containing your PDFs (project root directory)
PDF_FOLDER = os.getcwd()  # Current working directory

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.secret_key = 'enter your api key'

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB URI
db = client['user_db']  # Database name
users_collection = db['users']  # Collection for storing user data

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/pdf_table')
def pdf_table():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get patient_id from query parameters if provided
    patient_id = request.args.get('patient_id')
    
    # Build query based on user type and patient_id
    query = {}
    if session.get('user_type') == 'patient':
        # Patients can only see their own reports
        query['patient_id'] = ObjectId(session['user_id'])
    elif patient_id:
        # If patient_id is provided, show only that patient's reports
        query['patient_id'] = ObjectId(patient_id)
    
    # Get reports from database
    reports = list(db.reports.find(query).sort('created_at', -1))
    
    # Get patient info for each report
    files = []
    for report in reports:
        try:
            patient = users_collection.find_one({'_id': report['patient_id']})
            patient_name = patient['username'] if patient else 'Unknown Patient'
            
            files.append({
                'name': report['filename'],
                'url': url_for('view_report', filename=report['filename']),
                'date': report['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                'patient_name': patient_name,
                'doctor_name': report.get('doctor_name', 'Unknown Doctor'),
                'hospital_name': report.get('hospital_name', 'Unknown Hospital')
            })
        except Exception as e:
            print(f"Error processing report {report['_id']}: {str(e)}")
            continue
    
    return render_template('pdf_table.html', files=files)

@app.route("/view_report/<filename>")
def view_report(filename):
    if 'user_id' not in session:
        flash('Please log in to view reports', 'error')
        return redirect(url_for('login'))
    
    try:
        # Get the current user
        current_user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        
        # Get the report from database
        report = db.reports.find_one({'filename': filename})
        if not report:
            flash('Report not found', 'error')
            return redirect(url_for('pdf_table'))
        
        # Check access permissions
        if session.get('user_type') == 'patient' and str(report['patient_id']) != session['user_id']:
            flash('Access denied! You can only view your own reports.', 'error')
            return redirect(url_for('pdf_table'))
        
        # Serve the PDF from the project root directory
        return send_from_directory(os.getcwd(), filename)
    except Exception as e:
        flash(f'Error viewing report: {str(e)}', 'error')
        return redirect(url_for('pdf_table'))

@app.route('/record')
def record():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        session.pop('user_id', None)
        return redirect(url_for('login'))
    
    patient_id = request.args.get('patient_id')
    patient = None
    if patient_id:
        patient = users_collection.find_one({'_id': ObjectId(patient_id)})
        if not patient:
            flash('Patient not found', 'error')
            return redirect(url_for('hospital_dashboard'))
    
    # Get doctor's information
    doctor = users_collection.find_one({'_id': ObjectId(session['user_id'])})
    doctor_name = f"Dr. {doctor['username']}"
    hospital_name = doctor.get('hospital_name', 'Unknown Hospital')
    
    return render_template('record.html', 
                         patient=patient,
                         doctor_name=doctor_name,
                         hospital_name=hospital_name)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Handle form submission
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        phone_number = request.form.get('phone_number')
        age = request.form.get('age')
        gender = request.form.get('gender')

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('signup'))

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Check if the email or username already exists
        if users_collection.find_one({'email': email}):
            flash('Email is already registered!', 'error')
            return redirect(url_for('signup'))
        
        if users_collection.find_one({'username': username}):
            flash('Username is already taken!', 'error')
            return redirect(url_for('signup'))

        # Insert the new user into the database
        user_data = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'phone_number': phone_number,
            'age': int(age),
            'gender': gender,
            'user_type': 'patient'  # Add user type
        }
        users_collection.insert_one(user_data)
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/signup_hosp', methods=['POST','GET'])
def signup_hosp():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        hospital_name = request.form.get('hosp_name')

        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('index'))

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Check if the email or username already exists
        if users_collection.find_one({'email': email}):
            flash('Email is already registered!', 'error')
            return redirect(url_for('index'))
        
        if users_collection.find_one({'username': username}):
            flash('Username is already taken!', 'error')
            return redirect(url_for('index'))

        # Insert the new user into the database
        user_data = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'hospital_name': hospital_name,
            'user_type': 'hospital'  # Add user type
        }
        users_collection.insert_one(user_data)
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('index'))
    return render_template('signup_hosp.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Find the user by email
        user = users_collection.find_one({'email': email})
        
        if user and check_password_hash(user['password'], password):
            # Handle existing users without user_type
            if 'user_type' not in user:
                if 'hospital_name' in user:
                    user_type = 'hospital'
                else:
                    user_type = 'patient'
                # Update the user in database
                users_collection.update_one(
                    {'_id': user['_id']},
                    {'$set': {'user_type': user_type}}
                )
            else:
                user_type = user['user_type']
            
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            session['user_type'] = user_type
            
            if user_type == 'hospital':
                return redirect(url_for('hospital_dashboard'))
            else:
                return redirect(url_for('patient_dashboard'))
        else:
            flash('Invalid email or password!', 'error')
    return render_template('login.html')

@app.route('/hospital_dashboard')
def hospital_dashboard():
    if 'user_id' in session and session.get('user_type') == 'hospital':
        # Get all patients
        patients = list(users_collection.find({'user_type': 'patient'}))
        return render_template('hospital_dashboard.html', patients=patients)
    else:
        flash('Access denied! Please log in as a hospital.', 'error')
        return redirect(url_for('login'))

@app.route('/patient_dashboard')
def patient_dashboard():
    if 'user_id' in session and session.get('user_type') == 'patient':
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        
        # Get reports for this patient from database
        reports = list(db.reports.find({'patient_id': ObjectId(session['user_id'])}).sort('created_at', -1))
        
        files = []
        for report in reports:
            try:
                files.append({
                    "name": report['filename'],
                    "date": report['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                    "url": url_for('view_report', filename=report['filename']),
                    "doctor_name": report.get('doctor_name', 'Unknown Doctor'),
                    "hospital_name": report.get('hospital_name', 'Unknown Hospital')
                })
            except Exception as e:
                print(f"Error processing report {report['_id']}: {str(e)}")
                continue
        
        return render_template('patient_dashboard.html', user=user, files=files)
    else:
        flash('Access denied! Please log in as a patient.', 'error')
        return redirect(url_for('login'))

@app.route('/view_patient/<patient_id>')
def view_patient(patient_id):
    if 'user_id' in session and session.get('user_type') == 'hospital':
        patient = users_collection.find_one({'_id': ObjectId(patient_id), 'user_type': 'patient'})
        if patient:
            return render_template('view_patient.html', patient=patient)
        else:
            flash('Patient not found!', 'error')
            return redirect(url_for('hospital_dashboard'))
    else:
        flash('Access denied! Please log in as a hospital.', 'error')
        return redirect(url_for('login'))

@app.route('/main')
def main():
    # Check if the user is logged in
    if 'user_id' in session:
        return render_template('index.html', username=session['username'])
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    flash('You have been logged out!', 'success')
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'user_id' not in session:
        return jsonify({'error': 'Please log in to upload files'}), 401
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    patient_id = request.form.get('patient_id')
    if not patient_id:
        return jsonify({'error': 'No patient ID provided'}), 400
    
    # Get patient information
    patient = users_collection.find_one({'_id': ObjectId(patient_id)})
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    # Get doctor's information
    doctor_name = request.form.get('doctor_name', 'Unknown Doctor')
    hospital_name = request.form.get('hospital_name', 'Unknown Hospital')
    
    # Save the audio file
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(audio_path)
    
    try:
        # Process the audio file
        result = process_audio(audio_path)
        
        # Generate PDF report with modern styling
        pdf_filename = f"report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join(os.getcwd(), pdf_filename)
        
        # Generate the report using the modern styled function
        generate_reportpdf(
            patient_name=patient['username'],
            age=patient.get('age', 'N/A'),
            gender=patient.get('gender', 'N/A'),
            final_summary=result.get('summary', ''),
            file_name=pdf_path
        )
        
        # Create a record in the database for the report
        report_data = {
            'patient_id': ObjectId(patient_id),
            'doctor_id': ObjectId(session['user_id']),
            'filename': pdf_filename,
            'created_at': datetime.now(),
            'doctor_name': doctor_name,
            'hospital_name': hospital_name,
            'symptoms': result.get('symptoms', ''),
            'summary': result.get('summary', '')
        }
        
        # Add the report to the database
        db.reports.insert_one(report_data)
        
        # Clean up the audio file
        os.remove(audio_path)
        
        return jsonify({
            'message': 'File processed successfully',
            'pdf_url': url_for('view_report', filename=pdf_filename)
        })
        
    except Exception as e:
        # Clean up the audio file in case of error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({'error': str(e)}), 500

def transcribe_audio(audio_file_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio_file_path, return_timestamps=True)
    transcription = result["text"]
    
    return transcription

def speaker_map_transcription(transcription):
    client = OpenAI()

    instruction = f"""The below is the transcript of a conversation between a doctor and a patient, and the dialogues are not speaker mapped.
        You are instructed to read the whole conversation and identify the speaker's dialogues and label the dialogues as "Doctor:" and "Patient".
        Transcript:{transcription}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": instruction
        }],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    output = response.choices[0].message.content
    return output

import pandas as pd
from openai import OpenAI
from rapidfuzz import process

def extract_and_map_symptoms(output, csv_file):
    """
    Extract and map symptoms from a paragraph based on a list of known symptoms from a CSV file.

    Parameters:
    paragraph (str): The input text containing potential symptoms.
    csv_file (str): Path to the CSV file containing known symptoms.
    api_key (str): OpenAI API key.

    Returns:
    str: Mapped symptoms as a comma-separated string.
    """
    # Initialize OpenAI API
    #openai.api_key = api_key

    # Load symptoms from the CSV file
    def load_symptoms(csv_file):
        df = pd.read_csv(csv_file, encoding='latin-1')
        return list(df['Symptoms'].dropna().unique())

    # Normalize symptom strings
    def normalize_symptoms(symptoms):
        return [symptom.lower().strip() for symptom in symptoms]

    # Split grouped symptoms into individual ones
    def split_combined_symptoms(extracted_symptoms):
        split_symptoms = []
        for symptom in extracted_symptoms:
            # Split by common delimiters like newlines, commas, or dashes
            if '\n' in symptom or '-' in symptom or ',' in symptom:
                split_symptoms.extend(symptom.replace('-', '').replace('\n', ',').split(','))
            else:
                split_symptoms.append(symptom)
        return [sym.strip() for sym in split_symptoms if sym.strip()]  # Clean and remove empty strings

    # Map extracted symptoms to known symptoms using fuzzy matching
    def map_to_known_symptoms(extracted_symptoms, known_symptoms):
        mapped_symptoms = []
        for symptom in extracted_symptoms:
            # Find the best match from known symptoms
            best_match, score, _ = process.extractOne(symptom, known_symptoms)
            if score > 80:  # Only consider matches with a high similarity score
                mapped_symptoms.append(best_match)
        return mapped_symptoms

    # Load known symptoms
    known_symptoms = load_symptoms(csv_file)
    
    #paragraph=speaker_map_transcription(transcription)

    # Extract symptoms using OpenAI API
    prompt = f"Extract symptoms mentioned in the following text:\n\n{output}\n\nSymptoms:"
    client=OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.3
    )

    # Split response into individual symptoms
    extracted_symptoms = response.choices[0].message.content.strip().split(', ')
    extracted_symptoms = split_combined_symptoms(extracted_symptoms)  # Preprocess extracted symptoms
    normalized_extracted = normalize_symptoms(extracted_symptoms)
    normalized_known_symptoms = normalize_symptoms(known_symptoms)

    # Map extracted symptoms to standardized terms
    mapped_symptoms = map_to_known_symptoms(normalized_extracted, normalized_known_symptoms)

    # Return the mapped symptoms as a comma-separated string
    #return ", ".join(mapped_symptoms)
    string_symp= ", ".join(mapped_symptoms)
    symps=string_symp
    return symps

def summarized_conversation(mapped_output):
    client = OpenAI()

    instruction = f"""You are an Indian General Doctor , The below is the transcript of a conversation between a doctor and a patient, summarize the conversation . 
    Transcript:{mapped_output} , identify the symptoms present in the conversation , give the exact summarization of the conversation in the form of doctor's notes 
    and recommend medicines for the symptoms and generate a medical prescription with the name and brand of the general medicine along with dosage instructions and precautions.

    Example Output:

    Summary: Paste the conversation summary here

    Symptoms: Sore throat, fatigue, mild cough, slight fever (100 degrees)

    Possible Disease: Upper respiratory tract infection or viral infection
    
    Prescription:
    - Rest and hydration
    - Over-the-counter medications (acetaminophen, throat lozenges, cough syrup)
    - Consider throat swab for strep throat if indicated
    
    Dosage Instructions:
    - Acetaminophen (Brand: Tylenol): Take 500mg every 4-6 hours as needed for fever
    - Throat lozenges/sprays: Use as directed on packaging for sore throat
    - Cough syrup (Brand: Robitussin): Take 10ml every 4-6 hours as needed for cough
    
    Precautions:
    - Get plenty of rest
    - Stay hydrated
    - Avoid contact with sick individuals
    - Follow up if symptoms worsen or persist in the next few days
    
    Please consult with a healthcare professional before starting any new medications.

    The above example is the structure of the output you need to provide.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": instruction
        }],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    output_summary = response.choices[0].message.content
    return output_summary

def process_audio(audio_path):
    try:
        # Transcribe the audio
        transcription = transcribe_audio(audio_path)
        
        # Map speakers in the transcription
        mapped_output = speaker_map_transcription(transcription)
        
        # Extract and map symptoms
        symptoms = extract_and_map_symptoms(mapped_output, csv_file)
        
        # Generate summary and prescription
        final_summary = summarized_conversation(mapped_output)
        
        return {
            'transcription': transcription,
            'mapped_output': mapped_output,
            'symptoms': symptoms,
            'summary': final_summary
        }
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")

def generate_reportpdf(patient_name, age, gender, final_summary, file_name="patient_report.pdf"):
    # Create a PDF document template
    pdf = SimpleDocTemplate(file_name, pagesize=A4)
    width, height = A4

    # Define custom styles
    styles = getSampleStyleSheet()
    
    # Custom title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#2563eb'),
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    # Custom heading style
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=20,
        spaceBefore=20
    )
    
    # Custom normal style
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#1e293b'),
        leading=16,
        spaceAfter=12
    )
    
    # Custom patient info style
    patient_style = ParagraphStyle(
        'CustomPatient',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#1e293b'),
        leading=20,
        spaceAfter=20
    )

    # Create content
    elements = []

    # Add hospital logo and name
    elements.append(Paragraph("DOCTRAI", title_style))
    elements.append(Paragraph("Documentation and Optimization of Comprehensive Transcript Records with Assistive Intelligence", normal_style))
    
    # Add a decorative line
    elements.append(HRFlowable(
        width="100%",
        thickness=2,
        color=colors.HexColor('#2563eb'),
        spaceBefore=10,
        spaceAfter=20
    ))

    # Add patient details in a modern card-like format
    patient_details = f"""
    <b>Patient Information</b><br/>
    <font color='#64748b'>Name:</font> {patient_name}<br/>
    <font color='#64748b'>Age:</font> {age}<br/>
    <font color='#64748b'>Gender:</font> {gender}<br/>
    <font color='#64748b'>Date:</font> {datetime.now().strftime('%B %d, %Y')}
    """
    elements.append(Paragraph(patient_details, patient_style))
    
    # Add another decorative line
    elements.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor('#e2e8f0'),
        spaceBefore=10,
        spaceAfter=20
    ))

    # Add the "Medical Report" heading
    elements.append(Paragraph("Medical Report", heading_style))

    # Process the final summary to format it better
    summary_parts = final_summary.split('\n\n')
    for part in summary_parts:
        if part.strip():
            # Check if this is a section heading
            if ':' in part and len(part.split(':')[0]) < 30:
                section_title = part.split(':')[0]
                section_content = ':'.join(part.split(':')[1:])
                elements.append(Paragraph(f"<b>{section_title}:</b>", normal_style))
                elements.append(Paragraph(section_content.strip(), normal_style))
            else:
                elements.append(Paragraph(part, normal_style))
            elements.append(Spacer(1, 12))

    # Add footer
    footer = """
    <font color='#64748b' size=10>
    This report was generated by DOCTRAI Documentation and Optimization of Comprehensive Transcript Records with Assistive Intelligence.<br/>
    For any queries, please contact your healthcare provider.
    </font>
    """
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(footer, normal_style))

    # Build the PDF
    pdf.build(elements)

    return file_name

# Add this function to handle existing users
def migrate_existing_users():
    # Update users without user_type
    users_collection.update_many(
        {'user_type': {'$exists': False}},
        {'$set': {'user_type': 'patient'}}  # Default to patient for existing users
    )
    
    # Update hospital users based on hospital_name field
    users_collection.update_many(
        {'hospital_name': {'$exists': True}, 'user_type': 'patient'},
        {'$set': {'user_type': 'hospital'}}
    )

# Add this route to handle the migration
@app.route('/migrate')
def run_migration():
    try:
        migrate_existing_users()
        flash('Database migration completed successfully!', 'success')
    except Exception as e:
        flash(f'Migration failed: {str(e)}', 'error')
    return redirect(url_for('login'))

@app.route('/delete_patient/<patient_id>')
def delete_patient(patient_id):
    if 'user_id' not in session or session.get('user_type') != 'hospital':
        flash('Access denied! Only hospitals can delete patients.', 'error')
        return redirect(url_for('login'))
    
    try:
        # Verify the patient exists
        patient = users_collection.find_one({'_id': ObjectId(patient_id), 'user_type': 'patient'})
        if not patient:
            flash('Patient not found!', 'error')
            return redirect(url_for('hospital_dashboard'))
        
        # Delete the patient's reports
        pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
        for filename in pdf_files:
            if f'_{patient_id}.pdf' in filename:
                try:
                    os.remove(filename)
                except:
                    pass  # Continue even if file deletion fails
        
        # Delete the patient from the database
        users_collection.delete_one({'_id': ObjectId(patient_id)})
        
        flash('Patient deleted successfully!', 'success')
        return redirect(url_for('hospital_dashboard'))
    
    except Exception as e:
        flash(f'Error deleting patient: {str(e)}', 'error')
        return redirect(url_for('hospital_dashboard'))

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    if 'user_id' not in session:
        flash('Please log in to download reports', 'error')
        return redirect(url_for('login'))
    
    try:
        # Get the current user
        current_user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        
        # Extract patient ID from filename
        parts = filename.split('_')
        if len(parts) >= 4:
            patient_id = parts[-1].replace('.pdf', '')
            
            # If user is a patient, only allow downloading their own reports
            if session.get('user_type') == 'patient' and patient_id != str(current_user['_id']):
                flash('Access denied! You can only download your own reports.', 'error')
                return redirect(url_for('pdf_table'))
        
        # Serve the PDF from the uploads directory
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('pdf_table'))

if __name__ == '__main__':
    app.run(debug=True)