from flask import Flask, request, redirect, render_template
import pandas as pd
import joblib

# Declare a Flask app
app = Flask(__name__)

# Main function here
@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        tree = joblib.load("tree.pkl")
        
        # Unpickle classifier
        rfc = joblib.load("rfc.pkl")

        # Get values through input bars
        storage = int(request.form.get("storage"))
        ram = int(request.form.get("ram"))
        screenSize = float(request.form.get("screenSize"))
        frontCamera = int(request.form.get("frontCamera"))
        rearCamera = int(request.form.get("rearCamera"))
        battery = int(request.form.get("battery"))
        price = int(request.form.get("price"))

        # Put inputs to dataframe
        X = pd.DataFrame([[storage, ram, screenSize, frontCamera, rearCamera, battery, price]], columns = ["storage", "ram", "screenSize", "frontCamera", "rearCamera", "battery", "price"])
        
        # Get prediction
        pred_rfc = rfc.predict(X)[0]
        pred_tree = tree.predict(X)[0]
        
    else:
        pred_rfc = ""
        pred_tree = ""
        
    return render_template("index.html", output = [pred_rfc, pred_tree])

# Running the app
if __name__ == '__main__':
    app.run(debug = True)