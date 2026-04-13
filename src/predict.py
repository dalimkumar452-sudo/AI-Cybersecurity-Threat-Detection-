import joblib

def predict_sample(sample):
    model = joblib.load("models/model.pkl")
    prediction = model.predict([sample])

    if prediction[0] == 0:
        return "Normal Traffic"
    else:
        return "⚠️ Attack Detected!"