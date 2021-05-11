import pickle
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)


def diabetes_prediction(list_form):
    prediction = np.array(list_form).reshape(1, 8)
    model = pickle.load(open("model_diabetes.pkl", 'rb'))
    result = model.predict(prediction)

    return result[0]


@app.route('/')
def index():
    return render_template('index.html'), 200


@app.route('/prediction', methods=['POST',])
def prediction():
    if request.method == 'POST':
        form_list = request.form.to_dict()
        form_list = list(form_list.values())
        form_list = list(map(float, form_list))
        result = diabetes_prediction(form_list)

        if int(result) == 1:
            predict = 'Positive for Diabetes.'
        else:
            predict = 'Negative for Diabetes'

        return render_template('prediction.html', predict=predict)


app.run(debug=True)
