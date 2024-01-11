from flask import Flask, render_template, request
import pickle
from model import vectorizer  # Assuming vectorized is needed in your Flask app
app = Flask(__name__)
# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
@app.route('/')

def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        url = request.form['url']
        # Vectorize the input URL
        url_vectorized = vectorizer.transform([url])
        # Make a prediction using the trained model
        prediction = model.predict(url_vectorized)
        return render_template('result.html', url=url, result=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
