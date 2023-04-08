import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import models
import util

path = None
filename = None

UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/image')
def paintpath():
    global filename
    filename = "../static/css" + filename
    return render_template('image.html', path = filename)

@app.route('/result')
def result():
    global filename
    global path
    print(path)
    filename = "../static/css" + filename
    modelVGG = models.VGG19()
    modelVGG.take_image(path)
    predict1 = modelVGG.predict()
    predict1 = util.take_name_of_predict(predict1)
    modelDenseNet121 = models.DenseNet121()
    modelDenseNet121.take_image(path)
    predict2 = modelDenseNet121.predict()
    predict2 = util.take_name_of_predict(predict2)
    modelInceptionResNetV2 = models.InceptionResNetV2()
    modelInceptionResNetV2.take_image(path)
    predict3 = modelInceptionResNetV2.predict()
    predict3 = util.take_name_of_predict(predict3)
    return render_template('result.html', first = predict1, second = predict2, third = predict3)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Не могу прочитать файл')
            return redirect(request.url)
        file = request.files['file']
        # Если файл не выбран, то браузер может
        # отправить пустой файл без имени.
        if file.filename == '':
            flash('Нет выбранного файла')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            global filename
            filename = secure_filename(file.filename)
            global path
            path = os.path.join(app.config['UPLOAD_FOLDER'], 'static/css', filename)
            file.save(path)
            return redirect(url_for('result'))
    return render_template('index.html')

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

if __name__ == "__main__":
    app.run(debug=True)