# India Housing Investment Analyzer

A Streamlit portfolio app built with the cleaned India housing dataset.

## Features
- Sidebar filters for city, BHK, area, price, property type and availability
- KPI cards and interactive visual insights
- City/property-type price heatmap, city comparison, construction-year cohort chart, size-price scatter and amenity comparison
- Classification: "Good Investment" score
- Regression: estimated current price and a five-year appreciation scenario
- Classification validation metrics, regression metrics and feature importance
- Filtered-data preview and CSV download

## Important modelling assumptions
The source dataset has no verified investment outcome label and no historical five-year resale values.

1. **Classification proxy:** "Good Investment" is defined for this educational project as a Ready_to_Move property whose calculated price per square foot is at or below its city's median. The classifier learns this proxy label. Its probability is a model score for this proxy, not a guarantee of investment success.
2. **Five-year estimate:** a Random Forest regression model estimates the property's current price from its features. The app compounds that estimate using the annual appreciation rate selected by the user. Default is 5% per year; this is an assumption, not a forecast learned from historical price data.
3. **Price per sq. ft.:** recalculated as `Price_in_Lakhs * 100000 / Size_in_SqFt`, assuming the price column is in lakh INR.
4. Dataset values may be synthetic/randomized. Validate with verified transaction data before using the output for real decisions.

## Run locally

1. Install Python 3.10 or newer.
2. Put `india_housing_prices_cleaned(2).csv` in the same folder as `app.py` (the included CSV is already placed there), or upload it from the sidebar.
3. Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Model details
- Classification: Random Forest classifier; validation metrics include accuracy, precision, recall, F1 and ROC-AUC.
- Regression: Random Forest regressor; validation metrics include R², MAE and RMSE.
- Training uses a reproducible sample of up to 60,000 rows to keep the app responsive.
- Models are cached during the Streamlit session.
