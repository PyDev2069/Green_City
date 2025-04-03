from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ✅ Generate synthetic dataset
data = {
    "Green_Space": np.random.uniform(10, 80, 100),  # Green area in percentage
    "AQI": np.random.uniform(30, 200, 100),  # Air Quality Index
    "Population_Density": np.random.uniform(500, 10000, 100),  # People per km²
    "CO2_Emissions": np.random.uniform(1, 20, 100),  # CO₂ emissions in tons per capita
}

df = pd.DataFrame(data)

# ✅ Define labels (Green, Partially Green, Not Green)
conditions = [
    (df["Green_Space"] > 50) & (df["AQI"] < 70) & (df["CO2_Emissions"] < 5),
    (df["Green_Space"] > 30) & (df["AQI"] < 120) & (df["CO2_Emissions"] < 10),
    (df["Green_Space"] < 30) | (df["AQI"] > 120) | (df["CO2_Emissions"] > 10),
]
labels = ["Green City", "Partially Green City", "Not Green City"]
df["City_Type"] = np.select(conditions, labels, default="Not Green City")

# ✅ Train the model
X = df[["Green_Space", "AQI", "Population_Density", "CO2_Emissions"]]
y = df["City_Type"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# ✅ Web UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>City Green Classification</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        body {
            font-family: 'Poppins', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(-45deg, #228B22, #6A5ACD, #8B0000);
            background-size: 400% 400%;
            animation: gradientMove 10s ease infinite;
        }

        @keyframes gradientMove {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            width: 90%;
            max-width: 450px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.3);
            color: #fff;
            text-align: center;
            animation: floating 4s ease-in-out infinite;
        }

        @keyframes floating {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }

        h2 { color: #FFD700; }

        input {
            width: 90%;
            padding: 10px;
            margin: 10px 0;
            border: none;
            border-radius: 5px;
            text-align: center;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            box-shadow: 0px 0px 5px rgba(255, 255, 255, 0.3);
        }

        input::placeholder { color: rgba(255, 255, 255, 0.8); }

        input:focus {
            outline: none;
            box-shadow: 0 0 12px #FFD700;
            background: rgba(255, 255, 255, 0.3);
        }

        button {
            width: 100%;
            padding: 12px;
            background: #FFD700;
            color: black;
            font-size: 18px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease-in-out;
        }

        button:hover { background: #FFC107; }

        .result {
            font-size: 20px;
            margin-top: 20px;
            font-weight: bold;
            color: #FFD700;
            opacity: 0;
            transform: translateY(20px);
            animation: slideIn 0.5s ease-in-out forwards;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>🌍 City Green Classification 🌿</h2>
        <form action="/" method="post">
            <input type="number" step="0.01" name="green_space" placeholder="Green Space (%)" required><br>
            <input type="number" step="0.01" name="aqi" placeholder="Air Quality Index" required><br>
            <input type="number" step="0.01" name="density" placeholder="Population Density (people/km²)" required><br>
            <input type="number" step="0.01" name="co2" placeholder="CO₂ Emissions (tons per capita)" required><br>
            <button type="submit">Classify</button>
        </form>
        {% if result %}
            <div class="result">City Classification: <b>{{ result }}</b></div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def classify_city():
    result = None
    if request.method == "POST":
        try:
            user_data = np.array([[
                float(request.form["green_space"]),
                float(request.form["aqi"]),
                float(request.form["density"]),
                float(request.form["co2"])
            ]])
            scaled_data = scaler.transform(user_data)
            result = model.predict(scaled_data)[0]
        except ValueError:
            result = "Invalid input! Enter numeric values."

    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
