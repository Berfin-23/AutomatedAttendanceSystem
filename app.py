from flask import Flask, render_template, request, redirect, flash, url_for, jsonify
from firebase_admin import credentials, firestore, initialize_app
import os
import cv2
import numpy as np
import face_recognition
import base64
from PIL import Image
import io
import pickle
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/images/'
MODEL_PATH = 'model/face_recognition_model.pkl'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


cred = credentials.Certificate('./serviceAccountKey.json')
initialize_app(cred)
db = firestore.client()


def train_model():
    students_ref = db.collection('students').stream()
    known_face_encodings = []
    known_face_names = []

    for student in students_ref:
        student_data = student.to_dict()
        student_image_path = student_data['image_path']
        student_image = face_recognition.load_image_file(student_image_path)

        encodings = face_recognition.face_encodings(student_image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(student_data['name'])
            print(f"Loaded {student_data['name']} from {student_image_path}")

    model_data = {
        'encodings': known_face_encodings,
        'names': known_face_names
    }
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump(model_data, model_file)
    print("Model trained and saved successfully.")


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as model_file:
            model_data = pickle.load(model_file)
        return model_data
    else:
        return {'encodings': [], 'names': []}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        if 'image' in request.files:
            name = request.form['name']
            register_number = request.form['register_number']
            image = request.files['image']
            existing_students = db.collection('students').where('register_number', '==',
                                                                register_number.lower()).stream()
            if any(existing_student.id for existing_student in existing_students):
                flash('Register number already exists.')
                return redirect(url_for('admin'))

            if image:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{register_number}.png')
                image.save(image_path)
                db.collection('students').add({
                    'name': name,
                    'register_number': register_number.lower(),
                    'image_path': image_path
                })
                train_model()
                flash('Student uploaded and model trained!')
                return redirect(url_for('admin'))
        else:
            session_name = request.form['session_name']
            session_date = request.form['session_date']
            start_time = request.form['start_time']
            end_time = request.form['end_time']
            grace_time = request.form['grace_time']
            existing_sessions = db.collection('sessions').where('session_name', '==', session_name.lower()).stream()
            if any(existing_session.id for existing_session in existing_sessions):
                flash('Session name already exists.')
                return redirect(url_for('admin'))

            db.collection('sessions').add({
                'session_name': session_name,
                'date': session_date,
                'start_time': start_time,
                'end_time': end_time,
                'grace_time': grace_time
            })
            flash('Session added successfully!')
            return redirect(url_for('admin'))

    students_ref = db.collection('students').stream()
    students = [student.to_dict() for student in students_ref]
    sessions_ref = db.collection('sessions').stream()
    sessions = [session.to_dict() for session in sessions_ref]
    completed_sessions_ref = db.collection('completedSessions').stream()
    completed_sessions = [session.to_dict() for session in completed_sessions_ref]
    students_sorted = sorted(students, key=lambda x: int(x['register_number'][-4:]))
    sessions_sorted = sorted(sessions, key=lambda x: x['session_name'])
    completed_sessions_sorted = sorted(completed_sessions, key=lambda x: x['session_name'])

    return render_template('admin.html', students=students_sorted, sessions=sessions_sorted,
                           completed_sessions=completed_sessions_sorted)


@app.route('/get_attendance_details/<session_name>', methods=['GET'])
def get_attendance_details(session_name):
    completed_sessions_ref = db.collection('completedSessions').where('session_name', '==', session_name).stream()
    attendance_details = []
    for session in completed_sessions_ref:
        session_data = session.to_dict()
        attendance_details.extend(session_data.get('attendance', []))
    return jsonify(attendance_details)


@app.route('/attendance')
def attendance():
    sessions_ref = db.collection('sessions').stream()
    sessions = [session.to_dict() for session in sessions_ref]

    return render_template('attendance.html', sessions=sessions)


@app.route('/get_available_sessions', methods=['GET'])
def get_available_sessions():
    current_time = datetime.now()
    sessions_list = []

    # Fetch all sessions from Firestore
    sessions_ref = db.collection('sessions').stream()
    for session in sessions_ref:
        session_data = session.to_dict()
        session_start_time = datetime.strptime(f"{session_data['date']} {session_data['start_time']}", "%Y-%m-%d %H:%M")
        session_end_time = datetime.strptime(f"{session_data['date']} {session_data['end_time']}", "%Y-%m-%d %H:%M")

        # Determine if the session is selectable based on the current date and time
        is_selectable = (current_time < session_start_time - timedelta(minutes=15)) or \
                        (session_start_time <= current_time <= session_end_time)

        session_data['is_selectable'] = is_selectable
        sessions_list.append(session_data)

    return jsonify(sessions_list)


recognized_students_set = set()


@app.route('/recognize_faces', methods=['POST'])
def recognize_faces():
    global recognized_students_set

    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)

    model_data = load_model()
    known_face_encodings = model_data['encodings']
    known_face_names = model_data['names']

    rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    current_recognized_students = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            register_number = db.collection('students').where('name', '==', name).get()
            if register_number:
                register_number = register_number[0].to_dict()['register_number']

                current_recognized_students.append({
                    'name': name,
                    'register_number': register_number
                })
                recognized_students_set.add((register_number, name))
        else:
            current_recognized_students.append({
                'name': name,
                'register_number': "N/A"
            })

    all_recognized_students = [{'name': name, 'register_number': reg_no} for reg_no, name in recognized_students_set]

    return jsonify({
        'current_recognized_students': current_recognized_students,
        'all_recognized_students': all_recognized_students
    })


def get_arrival_status(current_time, session_start_time, grace_period):
    early_attendance_time = session_start_time - timedelta(minutes=15)
    grace_end_time = session_start_time + timedelta(minutes=grace_period)

    if early_attendance_time <= current_time <= grace_end_time:
        return "Present"
    elif current_time > grace_end_time:
        return "Late"
    return "Absent"


@app.route('/complete_session', methods=['POST'])
def complete_session():
    data = request.get_json()
    session = data['session']
    recognized_students = data['recognized_students']

    session_start_time = datetime.strptime(f"{session['date']} {session['start_time']}", "%Y-%m-%d %H:%M")
    grace_period = int(session['grace_time'])

    attendance_records = []
    for register_number in recognized_students:
        current_time = datetime.now()
        status = get_arrival_status(current_time, session_start_time, grace_period)
        attendance_records.append({
            'register_number': register_number,
            'attendance_time': current_time,
            'status': status
        })

    completed_session_data = {
        'session_name': session['session_name'],
        'date': session['date'],
        'start_time': session['start_time'],
        'end_time': session['end_time'],
        'grace_time': session['grace_time'],
        'attendance': attendance_records
    }

    db.collection('completedSessions').add(completed_session_data)

    sessions_ref = db.collection('sessions').where('session_name', '==', session['session_name']).stream()
    for session_doc in sessions_ref:
        db.collection('sessions').document(session_doc.id).delete()

    return jsonify({'message': 'Session completed successfully and moved to completedSessions'})


@app.route('/delete_student/<register_number>', methods=['POST'])
def delete_student(register_number):
    students_ref = db.collection('students').where('register_number', '==', register_number.lower()).stream()

    for student in students_ref:
        db.collection('students').document(student.id).delete()
        image_path = student.to_dict().get('image_path')
        if os.path.exists(image_path):
            os.remove(image_path)
    train_model()

    flash('Student deleted and model retrained successfully!')
    return redirect(url_for('admin'))


@app.route('/add_session', methods=['POST'])
def add_session():
    session_name = request.form['session_name']
    session_date = request.form['session_date']
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    grace_time = request.form['grace_time']

    existing_sessions = db.collection('sessions').where('session_name', '==', session_name.lower()).stream()
    if any(existing_session.id for existing_session in existing_sessions):
        flash('A session with this name already exists. Please use a different name.')
        return redirect(url_for('admin'))

    db.collection('sessions').add({
        'session_name': session_name,
        'date': session_date,
        'start_time': start_time,
        'end_time': end_time,
        'grace_time': grace_time
    })
    flash('Session added successfully!')
    return redirect(url_for('admin'))


@app.route('/delete_session/<session_name>', methods=['POST'])
def delete_session(session_name):
    sessions_ref = db.collection('sessions').where('session_name', '==', session_name).stream()

    session_found = False
    for session in sessions_ref:
        db.collection('sessions').document(session.id).delete()
        session_found = True

    if session_found:
        flash('Session deleted successfully!')
    else:
        flash('No session found with that name.')

    return redirect(url_for('admin'))


@app.route('/delete_completed_session/<session_name>', methods=['DELETE'])
def delete_completed_session(session_name):
    completed_sessions_ref = db.collection('completedSessions').where('session_name', '==', session_name).stream()
    session_found = False
    for session in completed_sessions_ref:
        db.collection('completedSessions').document(session.id).delete()
        session_found = True

    if session_found:
        return jsonify({'message': 'Session deleted successfully!'})
    else:
        return jsonify({'message': 'Session not found!'}), 404


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5511)
