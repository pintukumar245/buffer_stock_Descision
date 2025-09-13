from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf

app = Flask(__name__)

# ----------------------
# Load LSTM model & feature scaler
# ----------------------
model = tf.keras.models.load_model("C:/Users/pintu/OneDrive/Desktop/synthetic_prices.csv all data/ann.h5",compile=False)
scaler = pickle.load(open("C:/Users/pintu/OneDrive/Desktop/synthetic_prices.csv all data/scaler.pkl","rb"))

UPLOAD_FOLDER = "C:/Users/pintu/OneDrive/Desktop/synthetic_prices.csv all data/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------------
# Routes
# ----------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "⚠️ No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "⚠️ No file selected"

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Read CSV
    df = pd.read_csv(filepath)

    # Remove unnamed/extra columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # ----------------------
    # Features for model
    # ----------------------
    features = ["price","arrivals","sowing","rainfall","buffer_stock"]
    X = df[features].values

    # Scale features
    X_scaled = scaler.transform(X)

    # ----------------------
    # Create sequences (Sliding window / padding)
    # ----------------------
    sequences = []
    for i in range(len(X_scaled)):
        if i < 30:
            pad = np.repeat([X_scaled[0]], 30 - i, axis=0)
            seq = np.vstack((pad, X_scaled[:i+1]))
        else:
            seq = X_scaled[i-29:i+1]
        sequences.append(seq)

    X_seq = np.array(sequences)  # shape (num_samples, 30, 5)

    # ----------------------
    # Predict
    # ----------------------
    y_pred = model.predict(X_seq)

    # Round predicted price and original price
    df['predicted_price'] = np.round(y_pred.flatten(), 2)
    df['price'] = np.round(df['price'], 2)

    # ----------------------
    # Decision rule
    # ----------------------
    threshold = df['price'].mean()
    df['release'] = df['predicted_price'].apply(lambda x: "Yes" if x > threshold else "No")
    df['reason'] = df.apply(
        lambda row: "High Price, Release Needed" if row['predicted_price'] > threshold else "Normal Market, No Release",
        axis=1
    )

    # Save cleaned output CSV
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "predicted_output.csv")
    df.to_csv(output_path, index=False)

    # Show top 10 rows in HTML
    return render_template("result.html", tables=[df.head(10).to_html(classes='data', header="true")],
                           download_link="predicted_output.csv")

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

# ----------------------
# Run Flask
# ----------------------
if __name__ == "__main__":
    app.run(debug=True)
