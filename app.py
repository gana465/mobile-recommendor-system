from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# -------------------------------
# LOAD DATA
# -------------------------------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "2025_All_mobiles_Dataset.csv"))

df.columns = ['id','name','price','rating','ram','rom',
              'rear_cam','front_cam','battery','processor']

# -------------------------------
# CLEANING
# -------------------------------
df['price'] = df['price'].astype(str).str.replace(',', '').astype(float)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['ram'] = pd.to_numeric(df['ram'], errors='coerce')
df['rom'] = pd.to_numeric(df['rom'], errors='coerce')
df['battery'] = pd.to_numeric(df['battery'], errors='coerce')

df['battery'] = df['battery'].fillna(df['battery'].median())
df['processor'] = df['processor'].fillna("Unknown")

df = df.dropna(subset=['price','ram','rom','rating'])

df['brand'] = df['name'].apply(lambda x: x.split()[0].lower())

# -------------------------------
# ENCODING
# -------------------------------
le = LabelEncoder()
df['processor_encoded'] = le.fit_transform(df['processor'])

features = ['price','ram','rom','battery','processor_encoded']
X = df[features]
y = df['rating']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# MODELS
# -------------------------------
knn = KNeighborsRegressor().fit(X_scaled, y)
dt = DecisionTreeRegressor().fit(X_scaled, y)
rf = RandomForestRegressor().fit(X_scaled, y)

# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    brand = request.form.get('brand')
    price = request.form.get('price')

    if not brand or not price:
        return jsonify({"error": "Select brand and budget"})

    try:
        price = float(price)
    except:
        return jsonify({"error": "Budget must be number"})

    ram = request.form.get('ram')
    rom = request.form.get('rom')
    battery = request.form.get('battery')
    processor_input = request.form.get('processor')

    # -------------------------------
    # VALIDATION
    # -------------------------------
    if brand == "apple":
        if not rom:
            return jsonify({"error": "Enter ROM for Apple"})
        ram = float(rom)
        rom = float(rom)

    elif brand == "google":
        if not (ram and rom and battery):
            return jsonify({"error": "Enter RAM, ROM, Battery"})
        ram = float(ram)
        rom = float(rom)
        battery = float(battery)

    else:
        if not (ram and rom and battery and processor_input):
            return jsonify({"error": "Fill all Android fields"})
        ram = float(ram)
        rom = float(rom)
        battery = float(battery)

    # -------------------------------
    # PROCESSOR
    # -------------------------------
    processor = None
    if processor_input:
        matches = [x for x in le.classes_ if processor_input.lower() in x.lower()]
        if matches:
            processor = le.transform([matches[0]])[0]

    # -------------------------------
    # FILTER
    # -------------------------------
    filtered = filtered.dropna(subset=features)
    if brand == "apple":
        filtered = df[
            (df['price'] <= price * 1.02) &
            (df['name'].str.contains('iphone', case=False))
        ]

    elif brand == "google":
        filtered = df[
            (df['price'] <= price * 1.01) &
            (df['ram'] >= ram * 0.8) &
            (df['rom'] >= rom * 0.8) &
            (df['battery'] >= battery * 0.7) &
            (df['brand'].str.contains('google'))
        ]

    else:
        filtered = df[
            (df['price'] <= price * 1.01) &
            (df['ram'] >= ram * 0.7) &
            (df['rom'] >= rom * 0.5) &
            (df['battery'] >= battery * 0.6) &
            (~df['name'].str.contains('iphone', case=False)) &
            (~df['brand'].str.contains('google'))
        ]

        if processor is not None:
            selected = le.inverse_transform([processor])[0]
            temp = filtered[
                filtered['processor'].str.contains(selected, case=False)
            ]
            if not temp.empty:
                filtered = temp

    if filtered.empty:
        return jsonify({"error": "No mobiles found"})

    # -------------------------------
    # ML RANKING
    # -------------------------------
    X_f = scaler.transform(filtered[features])

    filtered = filtered.copy()
    filtered['score'] = (
        knn.predict(X_f) +
        dt.predict(X_f) +
        rf.predict(X_f)
    ) / 3

    top3 = filtered.sort_values(by='score', ascending=False).head(3)

    return jsonify(top3[['name','price','ram','rom']].to_dict(orient='records'))


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
