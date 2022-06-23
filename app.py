import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
import sys
app=Flask(__name__)
model5=pickle.load(open('model5.pkl','rb'))
@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def Predict():
    
    float_features = [float(x) for x in request.form.values()]
    print("Float Features: ",float_features,file=sys.stdout)
    #l=[0.373110,-0.362786,0.077662,0.263962,-0.543611]
    features = np.array(float_features).reshape(1,-1)
    print("Features:",features,file=sys.stdout)
    #l= np.asarray(l).reshape(1,-1)
    prediction = model5.predict(features)
    if prediction>=8:
        prediction="Excellent Quality" 
    elif prediction>=6.5 and prediction<8:
        prediction="Good Quality"
    else:
        prediction="Bad Quality"
    return render_template("index.html", prediction_text = "The wine quality is {}".format(prediction))




if __name__=="__main__":
    app.run(debug=True)

