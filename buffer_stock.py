import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from sklearn.metrics import mean_absolute_error,mean_squared_error



dataset = pd.read_csv(r"C:\Users\pintu\OneDrive\Desktop\synthetic_prices.csv all data\price_dataset_2025.csv",parse_dates=["date"])
dataset.set_index("date",inplace=True)

print(dataset.head(10))




# Features for model
features = ["price","arrivals","sowing","rainfall","buffer_stock"]
data = dataset[features].values


# Scaling
 
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

# Create Sequences
 
def create_sequences(data, seq_len=30):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])  
    return np.array(X), np.array(y)




SEQ_LEN = 30
X, y = create_sequences(scaled_data, SEQ_LEN)



# Train-test split
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]




# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)   
])






model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=50)

y_pred = model.predict(X_test)

print(y_pred)

# Inverse scaling
scale_price = scaler.scale_[0]
min_price = scaler.min_[0]


y_test_rescaled = y_test / scale_price - min_price/scale_price
y_pred_rescaled = y_pred.flatten() / scale_price - min_price/scale_price


me = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

print(me)
print(rse)


threshold = dataset["price"].mean() + 0.1 * df["price"].std()
buffer_stock = dataset["buffer_stock"].iloc[-len(y_test):].values.copy()
min_release = 500




decisions = []
for i in range(len(y_pred_rescaled)):
    if y_pred_rescaled[i] > threshold and buffer_stock[i] >= min_release:
        release = min_release
        buffer_stock[i] -= release
        reason = f"Predicted {y_pred_rescaled[i]:.2f} > threshold {threshold:.2f}"
    else:
        release = 200
        reason = "No trigger"
    decisions.append({"date": dataset.index[-len(y_test):][i], 
                      "predicted_price": y_pred_rescaled[i],
                      "release": release,
                      "reason": reason})
    
# how take a descision

decisions = []
for i in range(len(y_pred_rescaled)):
    if y_pred_rescaled[i] > threshold and buffer_stock[i] >= min_release:
        release = min_release
        buffer_stock[i] -= release
        reason = f"Predicted {y_pred_rescaled[i]:.2f} > threshold {threshold:.2f}"
    else:
        release = 200
        reason = "No trigger"
    decisions.append({"date": df.index[-len(y_test):][i], 
                      "predicted_price": y_pred_rescaled[i],
                      "release": release,
                      "reason": reason}) 
    
decision_df = pd.DataFrame(decisions)

print(decision_df)



plt.figure(figsize=(10,6))
plt.plot(df.index[-len(y_test):],y_test_rescaled,label="Actual Price")
plt.plot(df.index[-len(y_test):],y_pred_rescaled,label="Predicted Price")
plt.axhline(threshold,color="r",linestyle="--",label="Threshold")
plt.legend()
plt.show()