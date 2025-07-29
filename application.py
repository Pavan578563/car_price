from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
car=pd.read_csv('cleaned_car.csv')
selected_company=car['company'].unique().tolist()
name=car['name'].unique().tolist()
year=sorted(car['year'].unique().tolist(),reverse=True)
# Load the trained model
model = pickle.load(open('LinearRegression.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html',company=selected_company, name=name, year=year)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        company = request.form['company']
        year = int(request.form['year'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']

        data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        prediction = model.predict(data)[0]
        
        return render_template('index.html', prediction_text=f'Estimated Car Price: ₹ {int(prediction):,}')
if __name__=="__main__":
    app.run(debug=True)