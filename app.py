from flask import Flask, render_template, request
import os
from face_engine import load_students, recognize_faces

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_students()   # load student faces on startup


@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["photo"]
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    present_students = recognize_faces(path)

    return render_template("result.html", students=present_students)


if __name__ == "__main__":
    app.run(debug=True)
