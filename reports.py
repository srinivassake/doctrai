import os
from flask import Flask, render_template, send_from_directory, url_for
from datetime import datetime

app = Flask(__name__)

# Set the folder containing your PDFs (project root directory)
PDF_FOLDER = os.getcwd()  # Current working directory

@app.route("/")
def display_pdfs():
    # Get all PDF files in the root directory
    files = [
        {
            "name": f,
            "date": datetime.fromtimestamp(os.path.getmtime(os.path.join(PDF_FOLDER, f))).strftime('%Y-%m-%d %H:%M:%S'),
            "url": url_for('serve_pdf', filename=f)
        }
        for f in os.listdir(PDF_FOLDER)
        if f.endswith(".pdf")
    ]
    return render_template("pdf_table.html", files=files)

@app.route("/pdfs/<path:filename>")
def serve_pdf(filename):
    # Serve the PDF from the root directory
    return send_from_directory(PDF_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
