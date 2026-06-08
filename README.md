# Emirates Airways Reviews Dashboard

An interactive Streamlit dashboard for exploring Emirates Airways customer reviews, with built-in KPIs, charts, filters, and a regression model that predicts the Overall Rating.

## Features

- **CSV upload**: replace the dataset on the fly via a sidebar file uploader; all charts, KPIs, and the model recompute automatically
- **Filters**: narrow the data by Travel Type, Travel Class, Country, Recommended status, and Overall Rating range
- **KPIs**: review count, average overall rating, % recommended, average value-for-money, top travel class
- **Charts**: rating distribution, average rating by travel class, top countries by review count, recommended share, and a correlation heatmap of sub-ratings vs. overall rating
- **Prediction model**: a Linear Regression pipeline (median imputation for numeric sub-ratings, most-frequent imputation + one-hot encoding for categorical features) trained on an 80/20 split, reporting R², Adjusted R², MAE, and RMSE
- **Interactive prediction form**: enter feature values and get a live predicted Overall Rating

## Setup

```bash
cd Emirates_Dashboard
python3 -m pip install -r requirements.txt
```

## Run

```bash
streamlit run Emirates_Dashboard/app.py
```

The app opens in your browser at `http://localhost:8501`.

## Data

`Emirates Airways Reviews.csv` contains 1,540 customer reviews with sub-ratings (Seating Comfort, Staff Service, Food Quality, Entertainment, WiFi, Ground Service, Value for Money), trip metadata (Travel Type, Travel Class, Route, Country, Recommended), and an Overall Rating target. Free-text fields (Title, Review, dates, Route, Status) are excluded from the regression model.
