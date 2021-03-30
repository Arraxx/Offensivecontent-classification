import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Create a new Flask app.
app = Flask(__name__)


# API endpoint to classify feedbacks.
@app.route('/get_abuse_classification', methods=['POST'])
def get_abuse_classification():

	abuse_Classification = []
	json_data = request.json

	for data in json_data:
		tw = tokenizer.texts_to_sequences([data['blog']])
		tw = pad_sequences(tw, maxlen=200)
		abuse_Classification.append(str(model.predict_classes(tw)))

	return jsonify({'Abuse_Classification': str(abuse_Classification)})


	
if __name__ == '__main__':

	port = 8000


	with open('tokenizer1.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	with open('model1.json', 'r') as json_model_file:
		json_model = json_model_file.read()

	model = keras.models.model_from_json(json_model)
	model.load_weights('model1.h5')
	

	app.run(port=port, debug=True)



