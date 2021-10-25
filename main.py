import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app=Flask(__name__)
data=pd.read_csv("Cleaned_house_price.csv")
pipe=pickle.load(open("finalized_model.pickle",'rb'))


@app.route('/')
def index():
        return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Location = request.form.get('Location')
    BHK = int(request.form['BHK'])
    sqft = float(request.form['sqft'])
    floor = int(request.form['Floor'])
    
    print(Location, BHK, floor, sqft)
    input=pd.DataFrame([[Location,BHK,sqft,floor]],columns=['Location','BHK','Sq.ft','Floor'])
    prediction = pipe.predict(input)[0]

    return str(np.round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)