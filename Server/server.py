import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from models import getResults

path = None

UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/result')
def result():
    global path
    return render_template('result.html', data = getResults(path))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template('index.html', state = 'Не могу прочитать файл')
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', state = 'Нет выбранного файла')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                global path
                path = os.path.join(app.config['UPLOAD_FOLDER'], 'static/css', filename)
                file.save(path)
                return redirect(url_for('result'))
            else:
                return render_template('index.html', state = 'Неверный формат файла')
        return render_template('index.html')
    except RuntimeError as e:
        return render_template('index.html', state = 'Неизвестная ошибка')


app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

if __name__ == "__main__":
    app.run(debug=True)