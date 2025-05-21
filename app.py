from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import predict  # Import your modified prediction script

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'seceret'

# Path for saving uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Make prediction
        predicted_disease = predict.predict_image(file_path)
        return render_template('index.html', filename=filename, prediction=predicted_disease)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
