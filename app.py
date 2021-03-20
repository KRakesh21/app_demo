import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')

def home():
    return render_template('index.html')
    
@app.route('/predict', methods = ['POST'])

def predict():
    
     #it takes input form all text fields and stored it into int_features

    int_features = [int(x) for x in request.form.values()]
    
    #storeed the all values into array (to predict from model as all model required value or feature in np array format to predict)
    final_features = [np.array(int_features)]
   #predict the 
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    
    #render the data after prediction
    return render_template('index.html', prediction_text = 'Employee Salary should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug= True)
    
