import joblib
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import gunicorn

filename = 'modelsvm.pkl'
classifier = pickle.load(open(filename,'rb'))
model = pickle.load(open('modelsvm.pkl','rb'))

app = Flask(__name__,template_folder='Template')

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict-value', methods=['POST'])
def predict_value():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epithelial_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 4:
        res_val = "Malign ! The patient has a high risk of Breast Cancer !"
    else:
        res_val = "Benign ! The patient has Low risk of Breast cancer."
    return render_template('breastcancerresult.html', prediction_text = 'patient is{}'.format(res_val)) 

if __name__ == "__main__":
    app.run(debug=True)