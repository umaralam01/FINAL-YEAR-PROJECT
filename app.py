from flask import Flask, request, render_template, Response, send_file, redirect
import shutil
from flask import redirect
from sklearn.svm import SVC
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sqlite3

import joblib
from io import BytesIO, StringIO
import cv2
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import csv
import io
import base64
import matplotlib
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

matplotlib.use("agg")
import matplotlib.pyplot as plt
app = Flask(__name__)
ATTENDANCE_COOLDOWN = 30
MIN_WORK_TIME = 1
# ---------------- CONFIG ----------------
NIMGS = 50
DATETODAY = date.today().strftime("%d_%m_%y")
DATETODAY2 = date.today().strftime("%d-%B-%Y")

FACE_DETECTOR_PATH = "classifiers/haarcascade_frontalface_default.xml"
MODEL_PATH = "models/face_recognition_model.pkl"
METRICS_PATH = "models/metrics.pkl"
DB_PATH = f"Attendance/attendance_{DATETODAY}.db"
LOG_PATH = "logs/app.log"

face_detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

if face_detector.empty():
    raise Exception("Face detector not loaded.")

for d in ["Attendance", "static", "static/faces", "logs", "models"]:
    os.makedirs(d, exist_ok=True)

# ---------------- DB ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prenom TEXT NOT NULL,
    emp_id INTEGER NOT NULL,
    arrivee TEXT,
    depart TEXT,
    date TEXT NOT NULL
)
""")
conn.commit()

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_PATH,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- MODEL LOAD (GLOBAL FIX) ----------------
model = None
last_seen = {}
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logging.error(f"Model load failed: {e}")
        model = None

# ---------------- FUNCTIONS ----------------

def totalreg():
    if not os.path.exists("static/faces"):
        return 0

    return len(os.listdir("static/faces"))
#------
def get_registered_users():

    users = []

    for folder in os.listdir("static/faces"):

        if "_" in folder:

            name, emp_id = folder.rsplit("_", 1)

            users.append({
                "name": name,
                "id": emp_id
            })

    return users
#------
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.2, 5)

def identify_face(facearray):

    global model

    if model is None:
        return "Unknown"

    probs = model.predict_proba(facearray)[0]

    confidence = np.max(probs)

    if confidence < 0.65:
        return "Unknown"

    return model.predict(facearray)[0]
def train_model():
    logging.info("MODEL TRAINING STARTED")
    faces = []
    labels = []

    for user in os.listdir("static/faces"):
        for imgname in os.listdir(f"static/faces/{user}"):
            img = cv2.imread(f"static/faces/{user}/{imgname}")

            if img is None:
                continue

            img = preprocess_face(img)

            faces.append(
                img.flatten()
            )
            
            labels.append(user)

    if len(faces) == 0:
         return 

    faces = np.array(faces)
       
    X_train, X_test, y_train, y_test = train_test_split(
        faces, labels, test_size=0.2, random_state=42
    )

    svm = SVC(
    kernel="rbf",
    probability=True
    )
    svm.fit(
    X_train,
    y_train
    )

    y_pred = svm.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "matthews_cc": matthews_corrcoef(y_test, y_pred),
    }

    joblib.dump(metrics, METRICS_PATH)
    joblib.dump(
    svm,
    MODEL_PATH
    )
    logging.info(
    f"MODEL TRAINING COMPLETED | Samples: {len(faces)} | Users: {len(set(labels))}"
   )

    global model
    model = svm


def extract_attendance():
    c.execute(
        "SELECT prenom, emp_id, arrivee, depart FROM attendance WHERE date=?",
        (DATETODAY,),
    )
    rows = c.fetchall()

    return (
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        [r[3] for r in rows],
        len(rows),
    )

def present_count():
    c.execute(
        "SELECT COUNT(*) FROM attendance WHERE date=?",
        (DATETODAY,)
    )
    return c.fetchone()[0]


def absent_count():
    return max(0, totalreg() - present_count())

def should_mark_attendance(person):
    now = datetime.now()

    if person not in last_seen:
        last_seen[person] = now
        return True

    elapsed = (now - last_seen[person]).total_seconds()

    if elapsed > ATTENDANCE_COOLDOWN:
        last_seen[person] = now
        return True

    return False

def add_attendance(name):
    username, userid = name.split("_")[0], name.split("_")[1]
    time_now = datetime.now().strftime("%H:%M:%S")

    c.execute(
        "SELECT arrivee, depart FROM attendance WHERE date=? AND emp_id=?",
        (DATETODAY, userid),
    )
    row = c.fetchone()

    if row is None:

        c.execute(
            "INSERT INTO attendance (prenom, emp_id, arrivee, date) VALUES (?, ?, ?, ?)",
            (username, userid, time_now, DATETODAY)
        )
        logging.info(
                f"ARRIVAL | Employee: {username} | ID: {userid} | Time: {time_now} | Date: {DATETODAY}"
            )

    else:

        arrivee, depart = row

        if depart is None:

                arrival_time = datetime.strptime(arrivee, "%H:%M:%S")

                current_time = datetime.strptime(time_now, "%H:%M:%S")

                worked_minutes = (
                    current_time - arrival_time
                ).total_seconds() / 60

                if worked_minutes >= MIN_WORK_TIME:

                    c.execute(
                        "UPDATE attendance SET depart=? WHERE date=? AND emp_id=?",
                        (time_now, DATETODAY, userid)
                    )
                    logging.info(
                            f"DEPARTURE | Employee: {username} | ID: {userid} | Time: {time_now} | Date: {DATETODAY}"
                        )

    conn.commit()

    logging.info(
        f"Attendance marked for {username} ({userid})"
    )

#----------------help function
def get_status(arrivee, depart):

    if depart:
        return "Completed"

    return "Present"

def preprocess_face(face):

    gray = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    gray = cv2.resize(
        gray,
        (100,100)
    )

    return gray

# ---------------- VIDEO STREAM FIX ----------------
def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = extract_faces(frame)

        for (x, y, w, h) in faces:
            face = preprocess_face(
                    frame[y:y+h,x:x+w]
                )
            name = identify_face(face.reshape(1, -1))
            if name == "Unknown":

                logging.warning(
                    f"UNKNOWN FACE DETECTED | Time: {datetime.now().strftime('%H:%M:%S')}"
                )

            if name != "Unknown":
                if should_mark_attendance(name):
                       add_attendance(name)

            if name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                color,
                2
            )
            cv2.putText(
                    frame,
                    str(name),
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------- ROUTES ----------------

@app.route("/")
def home():

    names, rolls, arr, dep, l = extract_attendance()

    return render_template(
    "home.html",
    names=names,
    rolls=rolls,
    arrivees=arr,
    departs=dep,
    l=l,
    totalreg=totalreg(),
    present=present_count(),
    absent=absent_count(),
    datetoday2=DATETODAY2,
    users=get_registered_users()
)

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/metrics")
def metrics():
    if os.path.exists(METRICS_PATH):
        metrics = joblib.load(METRICS_PATH)
    else:
        metrics = {}

    return render_template("metrics.html", metrics=metrics)

@app.route("/delete/<emp_id>")
def delete_employee(emp_id):

    try:

        for folder in os.listdir("static/faces"):

            if folder.endswith(f"_{emp_id}"):

                shutil.rmtree(
                    os.path.join(
                        "static/faces",
                        folder
                    )
                )

                train_model()

                break

    except Exception as e:

        logging.error(str(e))

    return redirect("/")

@app.route("/add", methods=["GET", "POST"])
def add():
    newusername = request.form.get("newusername")
    newuserid = request.form.get("newuserid")
    logging.info(
    f"NEW EMPLOYEE REGISTERED | Name: {newusername} | ID: {newuserid}"
)
    logging.info(
    f"Cooldown started for {newusername}"
     )
    for existing in os.listdir("static/faces"):
        if existing.endswith(f"_{newuserid}"):
            return "Employee ID already exists"
    folder = f"static/faces/{newusername}_{newuserid}"
    os.makedirs(folder, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0

    try:

      while count < NIMGS:

        ret, frame = cap.read()

        if not ret:
            continue

        faces = extract_faces(frame)

        for (x, y, w, h) in faces:

            img = frame[y:y+h, x:x+w]

            cv2.imwrite(
                f"{folder}/{newusername}_{count}.jpg",
                img
            )

            count += 1

    finally:
          cap.release()
          
    train_model()

    return redirect("/")


@app.route("/export/csv")
def export_csv():
    names, rolls, arr, dep, _ = extract_attendance()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name", "ID", "Arrival", "Departure"])

    for i in zip(names, rolls, arr, dep):
        writer.writerow(i)

    return Response(output.getvalue(),
                    mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=attendance.csv"})


@app.route("/export/pdf")
def export_pdf():
    names, rolls, arr, dep, l = extract_attendance()

    buffer = BytesIO()
    c_pdf = canvas.Canvas(buffer, pagesize=letter)

    y = 750
    c_pdf.drawString(200, y, "Attendance Report")

    y -= 40
    for i in range(l):
        c_pdf.drawString(50, y, f"{names[i]} {rolls[i]} {arr[i]} {dep[i]}")
        y -= 20

    c_pdf.save()
    buffer.seek(0)

    return send_file(buffer,
                     mimetype='application/pdf',
                     as_attachment=True,
                     download_name="attendance.pdf")


if __name__ == "__main__":
    print("APP STARTING...")
    app.run(debug=True, host="0.0.0.0", port=5000)