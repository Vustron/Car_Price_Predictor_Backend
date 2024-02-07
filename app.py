from flask_cors import CORS,cross_origin
from flask import Flask, request
from flask import jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv('cleaned_car_data.csv')

@app.route('/data', methods=['GET', 'POST'])
@cross_origin()
def index():
    try:
        companies = car['company'].unique()
        car_models = car['name'].unique()
        year = car['year'].unique()
        fuel_type = car['fuel_type'].unique()
    
        data = {
            "companies": [str(company) for company in companies.tolist()],
            "car_models": [str(model) for model in car_models.tolist()],
            "years": [str(year) for year in year.tolist()],
            "fuel_types": [str(fuel) for fuel in fuel_type.tolist()]
        }

        return jsonify(data)
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    try:
        data = request.json
        companies = data.get('companies')
        car_models = data.get('car_models')
        years = data.get('years')
        fuel_types = data.get('fuel_types')
        kmTravelled = data.get('kmTravelled')

        prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'fuel_type', 'kms_driven',],
                                data=np.array([car_models,companies,years,fuel_types,kmTravelled]).reshape(1, 5)))
        
        response = {
            "prediction": np.round(prediction[0], 2).tolist()
        }

        return jsonify(response)
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
