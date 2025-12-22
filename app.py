from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# -------------------------
# Step 1: Load all models
# -------------------------
models = {
    "Naive Bayes": pickle.load(open("NB_model.pkl", 'rb')),
    "SVC": pickle.load(open("SVC_model.pkl", 'rb')),
   
    "Random Forest": pickle.load(open("RF_model.pkl", 'rb')),
    "Logistic Regression": pickle.load(open("LR_model.pkl", 'rb')),
    "KNN": pickle.load(open("KNN_model.pkl", 'rb')),
    "Decision Tree": pickle.load(open("DT_model.pkl", 'rb')),
    "Gradient Boosting": pickle.load(open("GBDT_model.pkl", 'rb')),
    "AdaBoost": pickle.load(open("Adaboost_model.pkl", 'rb')),
    "Bagging Classifier": pickle.load(open("Bgc_model.pkl", 'rb')),
}

vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

# -------------------------
# Step 2: Flask route
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        message = request.form["message"]
        selected_model = request.form["model"]

        # Transform the message
        text_vec = vectorizer.transform([message])

        # Make prediction
        model = models[selected_model]
        pred = model.predict(text_vec)[0]
        prediction = "🚫 Spam" if pred == 1 else "✅ Not Spam"

    return render_template("index.html", prediction=prediction, model_names=list(models.keys()))

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
