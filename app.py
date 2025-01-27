from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app=Flask(__name__)

classifier=pickle.load(open(r"C:\Users\khush\OneDrive\Documents\Desktop\ML_Projects\heart_model_prediction.pkl","rb"))


@app.route('/')

@app.route('/heart')
def home():
    return render_template("heart.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method=='POST':
        ChestPain=int(request.form["ChestPain"])
        BloodPressure=float(request.form["BloodPressure"])
        SerumCholestoral=int(request.form["SerumCholestoral"])
        BloodSugar=int(request.form["BloodSugar"])
        ECG=float(request.form["ECG"])
        HeartRate=float(request.form["HeartRate"])
        Angina=float(request.form["Angina"])


        input_data=(ChestPain,BloodPressure, SerumCholestoral, BloodSugar,ECG, HeartRate, Angina)
        input_data_as_numpy_array=np.asarray(input_data)
        input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
        prediction=classifier.predict(input_data_reshaped)

        if(prediction[0]==1):
            result="Sorry, you have chances of getting the disease. Please consult the doctor immediately."
        else:
            result="No need to fear. You have no dangerous symptoms of the disease."

        return render_template("result.html", result=result)
    

if __name__=='__main__':
    app.run(debug=True)
