from flask import Flask, render_template, request
import os
from vision_rag import explain_chart_with_rag

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None

    if request.method == "POST":
        question = request.form.get("question")
        image_file = request.files.get("image")

        if image_file and question:
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)

            answer = explain_chart_with_rag(image_path, question)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True) 