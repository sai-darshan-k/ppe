from flask import Flask, render_template, Response, jsonify, request, session, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
import cvzone
import math
import time
import threading
from playsound import playsound
from ultralytics import YOLO

# Flask App Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vinayak'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO Model
model = YOLO("best.pt")

# Define class names
classNames = [
    'Glass', 'Gloves', 'Goggles', 'Helmet', 'No-Helmet', 'No-Vest',
    'Person', 'Safety-Boot', 'Safety-Vest', 'Vest', 'helmet',
    'no helmet', 'no vest', 'no_helmet', 'no_vest',
    'protective_suit', 'vest', 'worker'
]

# Alert System
alert_playing = True
alert_lock = threading.Lock()


def play_alert():
    """Play alert sound in a loop."""
    while alert_playing:
        playsound("alert.mp3", block=False)
        time.sleep(1)  # Prevent overlapping sound


def trigger_alert(alert_required):
    """Start or stop the alert sound based on detection."""
    global alert_playing
    with alert_lock:
        if alert_required and not alert_playing:
            alert_playing = True
            threading.Thread(target=play_alert, daemon=True).start()
        elif not alert_required and alert_playing:
            alert_playing = False


# Flask Form
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")


def video_detection(input_source):
    """Perform object detection on video or webcam."""
    cap = cv2.VideoCapture(input_source)
    prev_frame_time = 0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        alert_required = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                label = f'{classNames[cls]} {conf}'
                cvzone.putTextRect(img, label, (x1, max(35, y1)), scale=1, thickness=1)

                if classNames[cls] in ['No-Helmet', 'No-Vest', 'no helmet', 'no vest', 'no_helmet', 'no_vest']:
                    alert_required = True

        trigger_alert(alert_required)

        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time else 0
        prev_frame_time = new_frame_time
        cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

        yield img

    cap.release()

def generate_frames(input_source):
    """Generate frames for Flask video streaming."""
    for frame in video_detection(input_source):
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    """Home page."""
    session.clear()
    return render_template('indexproject.html')


@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    """Webcam page."""
    session.clear()
    return render_template('ui.html')


@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    """Video upload page."""
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        session['video_path'] = file_path
        flash('Video uploaded successfully!', 'success')
    return render_template('videoprojectnew.html', form=form)


@app.route('/video')
def video():
    """Stream uploaded video."""
    video_path = session.get('video_path', None)
    if video_path:
        return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    flash('No video file uploaded!', 'error')
    return redirect('/video')


@app.route('/webapp')
def webapp():
    """Stream webcam video."""
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
