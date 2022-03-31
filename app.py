from flask import Flask, request, url_for, redirect, render_template, jsonify
#from pycaret.classification import load_model, predict_model
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# with open('/Users/sherapgyaltsen/Desktop/fish_forecast/knn_model.pkl', 'rb') as savefile:
#     model = pickle.load(savefile)
# with open('/Users/sherapgyaltsen/Desktop/fish_forecast/knn_model.pkl', 'rb') as f:
#     model = pickle.load(f)
#
#cols = ['Weight', 'Length1', 'Length2','Length3','Height','Width']

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
	int_features = [x for x in request.form.values()]
	final = np.array(int_features)
	final = final.astype(np.float64)
	#data_unseen = pd.DataFrame([final], columns=cols)
	model = pickle.load(open('notebook/logreg.pkl','rb'))
	prediction = model.predict([final])
	print(prediction)
	return render_template('home.html', prediction=prediction)

# @app.route('/predict', methods=['POST'])
# def predict():
# 	int_features = [x for x in request.form.values()]
# 	final = np.array(int_features)
# 	data_unseen = pd.DataFrame([final], columns=cols)
# 	prediction = predict_model(model, data = data_unseen, round=0)
# 	prediction = int(prediction.Label[0])
# 	return render_template('home.html', pred='Expected Fish Species will be {}'.format(prediction))

if __name__ == '__main__':
	app.run(debug=True)