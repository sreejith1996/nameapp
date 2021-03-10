from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
from model_files.ml_model import make_name
from flask_bootstrap import Bootstrap

app = Flask("dwarf_dino_generator")
Bootstrap(app)

@app.route('/')
def my_app():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def make_dwarf_name():
    data = request.form['dwarf_first_chars']
    loadingmodel = load_model('./model_files/my_model.h5')
    name = make_name(loadingmodel,data)
    return render_template('index.html', data=name)

#@app.route('/',methods=['GET'])
#def ping():
#   return "Pinging model application"

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)