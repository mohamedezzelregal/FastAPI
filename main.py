from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd

# Load model
model = joblib.load("hgb_best_model.joblib")

app = FastAPI()

# Home page with form
@app.get("/", response_class=HTMLResponse)
def form():
    return """
    <html>
        <head>
            <title>Housing Price Prediction</title>
        </head>
        <body>
            <h2>Enter Housing Data</h2>
            <form action="/predict" method="post">
                Longitude: <input type="number" step="any" name="longitude"><br>
                Latitude: <input type="number" step="any" name="latitude"><br>
                Housing Median Age: <input type="number" step="any" name="housing_median_age"><br>
                Total Rooms: <input type="number" step="any" name="total_rooms"><br>
                Total Bedrooms: <input type="number" step="any" name="total_bedrooms"><br>
                Population: <input type="number" step="any" name="population"><br>
                Households: <input type="number" step="any" name="households"><br>
                Median Income: <input type="number" step="any" name="median_income"><br>
                Ocean Proximity:
                <select name="ocean_proximity">
                    <option value="NEAR BAY">NEAR BAY</option>
                    <option value="INLAND">INLAND</option>
                    <option value="NEAR OCEAN">NEAR OCEAN</option>
                    <option value="ISLAND">ISLAND</option>
                    <option value="<1H OCEAN"><1H OCEAN</option>
                </select><br><br>

                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

# Handle form submission
@app.post("/predict", response_class=HTMLResponse)
def predict(
    longitude: float = Form(...),
    latitude: float = Form(...),
    housing_median_age: float = Form(...),
    total_rooms: float = Form(...),
    total_bedrooms: float = Form(...),
    population: float = Form(...),
    households: float = Form(...),
    median_income: float = Form(...),
    ocean_proximity: str = Form(...)
):
    # Convert to DataFrame
    data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])

    # Predict
    prediction = model.predict(data)[0]

    return f"""
    <html>
        <body>
            <h2>Predicted House Value: {prediction:,.2f}</h2>
            <a href="/">Try again</a>
        </body>
    </html>
    """
