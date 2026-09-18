

import joblib
import os


def load_artifacts():
    if not (os.path.exists("spam_model.pkl") and os.path.exists("spam_preprocessor.pkl")):
        raise FileNotFoundError(
            "Model files not found. Run 'python train.py' first to train "
            "and save the model."
        )
    model = joblib.load("spam_model.pkl")
    preprocessor = joblib.load("spam_preprocessor.pkl")
    return model, preprocessor


def predict_message(message, model, preprocessor):
    vec = preprocessor.transform([message])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]  # [prob_ham, prob_spam]

    label = "SPAM" if prediction == 1 else "NOT SPAM"
    confidence = probability[prediction] * 100
    return label, confidence


if __name__ == "__main__":
    print("Loading model...")
    model, preprocessor = load_artifacts()
    print("Model loaded! Type a message to check if it's spam.")
    print("(type 'exit' to quit)\n")

    while True:
        message = input("Enter message: ")
        if message.strip().lower() == "exit":
            break
        if not message.strip():
            continue

        label, confidence = predict_message(message, model, preprocessor)
        print(f"--> {label}  (confidence: {confidence:.1f}%)\n")