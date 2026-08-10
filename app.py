import os
import io
import base64
import json
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from agent.analyzer import analyze_dataset
from agent.model_selector import run_model_selection

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "Please upload a CSV file."}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Only CSV files are allowed."}), 400

    try:
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({"error": "The CSV file is empty."}), 400
        if len(df) < 10:
            return jsonify({"error": "Dataset must contain at least 10 rows."}), 400

        info = analyze_dataset(df)
        return jsonify(info)
    except Exception as exc:
        return jsonify({"error": f"Could not analyze dataset: {exc}"}), 400

@app.route("/train", methods=["POST"])
def train():
    if "file" not in request.files:
        return jsonify({"error": "Please upload a CSV file again."}), 400

    target = request.form.get("target", "").strip()
    file = request.files["file"]

    if not target:
        return jsonify({"error": "Please select a target column."}), 400
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Only CSV files are allowed."}), 400

    try:
        df = pd.read_csv(file)
        if target not in df.columns:
            return jsonify({"error": "Selected target column was not found."}), 400

        result = run_model_selection(df, target)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.route("/download", methods=["POST"])
def download():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No results supplied."}), 400

    report = io.StringIO()
    report.write("AI/ML Model Selection Agent Report\n")
    report.write("=" * 40 + "\n\n")
    report.write(f"Problem Type: {data.get('problem_type', '')}\n")
    report.write(f"Target Column: {data.get('target_column', '')}\n")
    report.write(f"Recommended Model: {data.get('best_model', '')}\n")
    report.write(f"Reason: {data.get('reason', '')}\n\n")
    report.write("Model Comparison\n")
    report.write("-" * 40 + "\n")

    results = data.get("results", [])
    if results:
        report.write(", ".join(results[0].keys()) + "\n")
        for row in results:
            report.write(", ".join(str(row.get(k, "")) for k in results[0].keys()) + "\n")

    output = io.BytesIO(report.getvalue().encode("utf-8"))
    output.seek(0)
    return send_file(output, as_attachment=True,
                     download_name="model_selection_report.txt",
                     mimetype="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
