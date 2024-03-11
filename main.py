from flask import Flask,render_template,request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv', index_col=0)

pipe=pickle.load(open('RidgeModel.pkl','rb'))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template('index.html',locations=locations)


@app.route('/predict',methods=['POST'])
def predict():
    locations=request.form.get('location')
    bhk=float(request.form.get('bhk'))
    bathroom=float(request.form.get('bathroom'))
    square_feet=float(request.form.get('square feet'))
    df=pd.DataFrame([[locations,square_feet,bathroom,bhk]],columns=['location','total_sqft','bath','BHK'])
    prediction=pipe.predict(df)[0]
    return str(round((prediction*1000000),2))


if __name__ == '__main__':
    app.run(debug=True,port=5000)
