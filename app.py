import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# import model.py file having finial model details
import model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    user_input = request.form['txtarea']

    # First user recommendation

    recom_prods = model.recommendation(user_input)
    
    # Sentiment model

    output = model.sentiment(recom_prods)

    return render_template('index.html', tables=[output.to_html(classes='data',index=False)])


if __name__ == "__main__":
    app.run(debug=True)