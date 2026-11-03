import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)

st.set_page_config(
    page_title="India Housing Investment Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏠 India Housing Investment Analyzer")
st.caption("Explore property listings, estimate current value, and test a five-year appreciation scenario.")

st.info(
    "Important modelling note: this dataset does not contain a verified 'good investment' label "
    "or historical five-year sale prices. The classification below is a **proxy**: a property is "
    "labelled 'Good Investment' when it is Ready_to_Move and its calculated price per sq. ft. is "
    "at or below its city's median. The five-year value is a current-price model estimate grown "
    "using the appreciation assumption you select. It is a scenario, not a guaranteed forecast."
)

DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "india_housing_prices_cleaned.csv")

@st.cache_data(show_spinner="Loading housing dataset...")
def load_data(file_bytes=None):
    if file_bytes is not None:
        from io import BytesIO
        data = pd.read_csv(BytesIO(file_bytes))
    elif os.path.exists(DEFAULT_FILE):
        data = pd.read_csv(DEFAULT_FILE)
    else:
        return None

    # Normalize expected numeric columns
    numeric_cols = [
        "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No",
        "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "Price_per_SqFt" not in data.columns or data["Price_per_SqFt"].isna().all():
        data["Price_per_SqFt"] = np.nan
    # Recalculate consistently: Price_in_Lakhs is assumed to be lakh INR.
    if {"Price_in_Lakhs", "Size_in_SqFt"}.issubset(data.columns):
        data["Price_per_SqFt_Calculated"] = (
            data["Price_in_Lakhs"] * 100000 / data["Size_in_SqFt"].replace(0, np.nan)
        )
    return data

uploaded = st.sidebar.file_uploader(
    "Upload cleaned housing CSV (optional)",
    type=["csv"],
    help="If the CSV is in the same folder as app.py, it loads automatically."
)
df = load_data(uploaded.getvalue() if uploaded else None)

if df is None:
    st.warning("Place india_housing_prices_cleaned(2).csv beside app.py, or upload the cleaned CSV in the sidebar.")
    st.stop()

required = [
    "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
    "Price_in_Lakhs", "Furnished_Status", "Year_Built", "Floor_No",
    "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security",
    "Amenities", "Facing", "Owner_Type", "Availability_Status"
]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
    st.stop()

df = df.dropna(subset=["Price_in_Lakhs", "Size_in_SqFt", "BHK", "City"]).copy()
df = df[(df["Size_in_SqFt"] > 0) & (df["Price_in_Lakhs"] > 0)]
df["Price_per_SqFt_Calculated"] = df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
df["Property_Age"] = (2026 - df["Year_Built"]).clip(lower=0)

# -------------------- Sidebar filters --------------------
st.sidebar.header("🔎 Property Filters")
cities = sorted(df["City"].dropna().astype(str).unique())
selected_cities = st.sidebar.multiselect("City", cities, default=[])
bhk_values = sorted(df["BHK"].dropna().astype(int).unique())
selected_bhk = st.sidebar.multiselect("BHK", bhk_values, default=[])
min_area, max_area = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())
area_range = st.sidebar.slider("Area (sq. ft.)", min_area, max_area, (min_area, max_area))
min_price, max_price = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
price_range = st.sidebar.slider(
    "Price (₹ lakhs)", float(np.floor(min_price)), float(np.ceil(max_price)),
    (float(np.floor(min_price)), float(np.ceil(max_price)))
)
property_types = sorted(df["Property_Type"].dropna().astype(str).unique())
selected_types = st.sidebar.multiselect("Property type", property_types, default=[])
availability_options = sorted(df["Availability_Status"].dropna().astype(str).unique())
selected_availability = st.sidebar.multiselect("Availability", availability_options, default=[])

filtered = df.copy()
if selected_cities:
    filtered = filtered[filtered["City"].astype(str).isin(selected_cities)]
if selected_bhk:
    filtered = filtered[filtered["BHK"].isin(selected_bhk)]
if selected_types:
    filtered = filtered[filtered["Property_Type"].astype(str).isin(selected_types)]
if selected_availability:
    filtered = filtered[filtered["Availability_Status"].astype(str).isin(selected_availability)]
filtered = filtered[
    filtered["Size_in_SqFt"].between(area_range[0], area_range[1]) &
    filtered["Price_in_Lakhs"].between(price_range[0], price_range[1])
]

# -------------------- Overview --------------------
st.subheader("📊 Housing Market Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Properties in filter", f"{len(filtered):,}")
k2.metric("Average price", f"₹{filtered['Price_in_Lakhs'].mean():,.2f} L" if len(filtered) else "—")
k3.metric("Average area", f"{filtered['Size_in_SqFt'].mean():,.0f} sq. ft." if len(filtered) else "—")
k4.metric("Average price / sq. ft.", f"₹{filtered['Price_per_SqFt_Calculated'].mean():,.0f}" if len(filtered) else "—")

tab1, tab2, tab3, tab4 = st.tabs([
    "Market Insights", "Investment Assessment", "Model Performance", "Data Preview"
])

with tab1:
    if filtered.empty:
        st.warning("No properties match the selected filters. Widen the filters to see results.")
    else:
        left, right = st.columns(2)
        city_avg = (filtered.groupby("City", as_index=False)
                    .agg(Average_Price=("Price_in_Lakhs", "mean"),
                         Property_Count=("Price_in_Lakhs", "size"))
                    .sort_values("Average_Price", ascending=False).head(15))
        left.plotly_chart(
            px.bar(city_avg, x="City", y="Average_Price", color="Average_Price",
                   hover_data=["Property_Count"], title="Top 15 Cities by Average Property Price",
                   labels={"Average_Price": "Average price (₹ lakhs)"}),
            use_container_width=True
        )
        type_avg = (filtered.groupby("Property_Type", as_index=False)["Price_in_Lakhs"]
                    .mean().sort_values("Price_in_Lakhs", ascending=False))
        right.plotly_chart(
            px.bar(type_avg, x="Property_Type", y="Price_in_Lakhs", color="Property_Type",
                   title="Average Price by Property Type",
                   labels={"Price_in_Lakhs": "Average price (₹ lakhs)"}),
            use_container_width=True
        )

        left, right = st.columns(2)
        heat = (filtered[filtered["City"].isin(city_avg["City"])]
                .pivot_table(index="City", columns="Property_Type",
                             values="Price_in_Lakhs", aggfunc="mean"))
        left.plotly_chart(
            px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                      title="Location-wise Price Heatmap (City × Property Type)",
                      labels=dict(color="Avg price (₹ lakhs)")),
            use_container_width=True
        )
        cohort = (filtered.groupby("Year_Built", as_index=False)["Price_in_Lakhs"]
                  .mean().sort_values("Year_Built"))
        right.plotly_chart(
            px.line(cohort, x="Year_Built", y="Price_in_Lakhs", markers=True,
                    title="Average Price by Construction Year",
                    labels={"Year_Built": "Year built", "Price_in_Lakhs": "Average price (₹ lakhs)"}),
            use_container_width=True
        )

        left, right = st.columns(2)
        left.plotly_chart(
            px.scatter(filtered.sample(min(10000, len(filtered)), random_state=42),
                       x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type",
                       hover_data=["City", "BHK"], opacity=0.55,
                       title="Property Size vs Price",
                       labels={"Size_in_SqFt": "Area (sq. ft.)", "Price_in_Lakhs": "Price (₹ lakhs)"}),
            use_container_width=True
        )
        amenity = (filtered.assign(Amenity=filtered["Amenities"].fillna("").str.split(", "))
                   .explode("Amenity"))
        amenity = amenity[amenity["Amenity"].ne("")]
        amenity_avg = (amenity.groupby("Amenity", as_index=False)["Price_in_Lakhs"]
                       .mean().sort_values("Price_in_Lakhs", ascending=False))
        right.plotly_chart(
            px.bar(amenity_avg, x="Amenity", y="Price_in_Lakhs",
                   title="Average Price by Amenity",
                   labels={"Price_in_Lakhs": "Average price (₹ lakhs)"}),
            use_container_width=True
        )

        st.markdown("**Interpretation note:** the construction-year chart compares property cohorts; it is not a time-series record of annual market prices.")

# -------------------- Modelling --------------------
numeric_features = [
    "BHK", "Size_in_SqFt", "Property_Age", "Floor_No", "Total_Floors",
    "Nearby_Schools", "Nearby_Hospitals"
]
categorical_features = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Parking_Space", "Security",
    "Amenities", "Facing", "Owner_Type", "Availability_Status"
]
feature_cols = numeric_features + categorical_features

@st.cache_resource(show_spinner="Training classification and price-estimation models (first run only)...")
def train_models(data):
    work = data.copy()
    work["Property_Age"] = (2026 - work["Year_Built"]).clip(lower=0)
    work["Price_per_SqFt_Calculated"] = (
        work["Price_in_Lakhs"] * 100000 / work["Size_in_SqFt"].replace(0, np.nan)
    )
    city_medians = work.groupby("City")["Price_per_SqFt_Calculated"].median()
    work["City_Median_PPSF"] = work["City"].map(city_medians)
    work["Good_Investment_Proxy"] = (
        (work["Availability_Status"].astype(str) == "Ready_to_Move") &
        (work["Price_per_SqFt_Calculated"] <= work["City_Median_PPSF"])
    ).astype(int)

    # Keep model training responsive while preserving a reproducible sample.
    sample_n = min(60000, len(work))
    work = work.sample(sample_n, random_state=42) if len(work) > sample_n else work
    X = work[feature_cols].copy()
    for c in categorical_features:
        X[c] = X[c].fillna("Unknown").astype(str)
    for c in numeric_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    y_class = work["Good_Investment_Proxy"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    pre_class = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categorical_features)
    ])
    classifier = Pipeline([
        ("prep", pre_class),
        ("model", RandomForestClassifier(
            n_estimators=130, max_depth=16, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ))
    ])
    classifier.fit(Xc_train, yc_train)
    class_pred = classifier.predict(Xc_test)
    class_prob = classifier.predict_proba(Xc_test)[:, 1]
    class_metrics = {
        "accuracy": accuracy_score(yc_test, class_pred),
        "precision": precision_score(yc_test, class_pred, zero_division=0),
        "recall": recall_score(yc_test, class_pred, zero_division=0),
        "f1": f1_score(yc_test, class_pred, zero_division=0),
        "auc": roc_auc_score(yc_test, class_prob) if yc_test.nunique() == 2 else np.nan
    }

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, work["Price_in_Lakhs"], test_size=0.2, random_state=42
    )
    pre_reg = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categorical_features)
    ])
    regressor = Pipeline([
        ("prep", pre_reg),
        ("model", RandomForestRegressor(
            n_estimators=130, max_depth=18, min_samples_leaf=3,
            max_features=0.85, random_state=42, n_jobs=-1
        ))
    ])
    regressor.fit(Xr_train, yr_train)
    reg_pred = regressor.predict(Xr_test)
    abs_errors = np.abs(yr_test.to_numpy() - reg_pred)
    reg_metrics = {
        "mae": mean_absolute_error(yr_test, reg_pred),
        "rmse": np.sqrt(mean_squared_error(yr_test, reg_pred)),
        "r2": r2_score(yr_test, reg_pred),
        "error90": float(np.quantile(abs_errors, 0.90))
    }

    # Use classification feature importance for the investment proxy and regression
    # feature importance for estimated price; show the regression model's importance.
    names = regressor.named_steps["prep"].get_feature_names_out()
    importances = regressor.named_steps["model"].feature_importances_
    importance_df = (pd.DataFrame({"Feature": names, "Importance": importances})
                     .sort_values("Importance", ascending=False).head(20))
    return classifier, regressor, class_metrics, reg_metrics, importance_df, city_medians

with tab2:
    st.subheader("🧮 Enter a Property to Assess")
    st.write("Complete the property details and select **Analyze property**.")
    # Use dataset values as safe defaults.
    def options_for(col):
        vals = sorted(df[col].dropna().astype(str).unique())
        return vals if vals else ["Unknown"]

    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            input_state = st.selectbox("State", options_for("State"))
            input_city = st.selectbox("City", options_for("City"))
            input_locality = st.selectbox("Locality", options_for("Locality"))
            input_type = st.selectbox("Property type", options_for("Property_Type"))
            input_bhk = st.selectbox("BHK", bhk_values, index=min(2, len(bhk_values)-1))
            input_area = st.number_input("Area (sq. ft.)", min_value=100, max_value=50000, value=1500, step=50)
        with c2:
            input_furnish = st.selectbox("Furnishing", options_for("Furnished_Status"))
            input_year = st.number_input("Year built", min_value=1950, max_value=2026, value=2015, step=1)
            input_floor = st.number_input("Floor number", min_value=0, max_value=100, value=2, step=1)
            input_total_floors = st.number_input("Total floors", min_value=1, max_value=100, value=5, step=1)
            input_schools = st.number_input("Nearby schools", min_value=0, max_value=30, value=3, step=1)
            input_hospitals = st.number_input("Nearby hospitals", min_value=0, max_value=30, value=2, step=1)
        with c3:
            input_transport = st.selectbox("Public transport accessibility", options_for("Public_Transport_Accessibility"))
            input_parking = st.selectbox("Parking space", options_for("Parking_Space"))
            input_security = st.selectbox("Security", options_for("Security"))
            input_amenities = st.selectbox("Amenities", options_for("Amenities"))
            input_facing = st.selectbox("Facing", options_for("Facing"))
            input_owner = st.selectbox("Owner type", options_for("Owner_Type"))
            input_availability = st.selectbox("Availability", options_for("Availability_Status"))
        appreciation = st.slider("Assumed annual appreciation for next 5 years (%)", 0.0, 15.0, 5.0, 0.5)
        submitted = st.form_submit_button("Analyze property", type="primary", use_container_width=True)

    if submitted:
        if input_total_floors < input_floor:
            st.error("Total floors should be greater than or equal to the selected floor number.")
        else:
            with st.spinner("Preparing model estimates..."):
                classifier, regressor, class_metrics, reg_metrics, importance_df, city_medians = train_models(df)
                record = {
                    "State": input_state, "City": input_city, "Locality": input_locality,
                    "Property_Type": input_type, "BHK": int(input_bhk),
                    "Size_in_SqFt": float(input_area), "Property_Age": max(0, 2026-int(input_year)),
                    "Floor_No": int(input_floor), "Total_Floors": int(input_total_floors),
                    "Nearby_Schools": int(input_schools), "Nearby_Hospitals": int(input_hospitals),
                    "Furnished_Status": input_furnish,
                    "Public_Transport_Accessibility": input_transport,
                    "Parking_Space": input_parking, "Security": input_security,
                    "Amenities": input_amenities, "Facing": input_facing,
                    "Owner_Type": input_owner, "Availability_Status": input_availability
                }
                input_df = pd.DataFrame([record], columns=feature_cols)
                current_est = float(regressor.predict(input_df)[0])
                annual_factor = 1 + appreciation / 100
                future_est = current_est * (annual_factor ** 5)
                error90 = reg_metrics["error90"]
                current_low = max(0, current_est - error90)
                current_high = current_est + error90
                future_low = current_low * (annual_factor ** 5)
                future_high = current_high * (annual_factor ** 5)
                probability = float(classifier.predict_proba(input_df)[0][1])
                decision = int(probability >= 0.5)

            st.divider()
            st.subheader("Assessment Results")
            r1, r2, r3 = st.columns(3)
            r1.metric("Investment assessment", "Good Investment" if decision else "Needs Caution")
            r2.metric("Model score for Good Investment", f"{probability*100:.1f}%")
            r3.metric("Estimated current price", f"₹{current_est:,.2f} lakhs")
            st.metric("Estimated price after 5 years", f"₹{future_est:,.2f} lakhs",
                      help=f"Assumes {appreciation:.1f}% annual appreciation compounded for five years.")
            st.caption(
                f"Illustrative 90% residual-error band: ₹{future_low:,.2f}L–₹{future_high:,.2f}L. "
                "This is based on held-out model errors, not a formal statistical prediction interval."
            )
            if decision:
                st.success("The model assigns this property a higher score for the dataset-defined investment proxy.")
            else:
                st.warning("The model assigns this property a lower score for the dataset-defined investment proxy.")
            st.write(
                f"**Five-year scenario:** current model estimate × (1 + {appreciation:.1f}%)⁵. "
                "Change the appreciation assumption to compare scenarios."
            )

with tab3:
    st.subheader("📈 Model Performance & Feature Importance")
    if st.button("Train / Load Models", type="primary"):
        with st.spinner("Training models..."):
            classifier, regressor, class_metrics, reg_metrics, importance_df, city_medians = train_models(df)
        st.session_state["models_ready"] = True
        st.session_state["class_metrics"] = class_metrics
        st.session_state["reg_metrics"] = reg_metrics
        st.session_state["importance_df"] = importance_df

    if st.session_state.get("models_ready"):
        cm = st.session_state["class_metrics"]
        rm = st.session_state["reg_metrics"]
        st.markdown("**Classification – Good Investment proxy**")
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{cm['accuracy']:.3f}")
        b.metric("Precision", f"{cm['precision']:.3f}")
        c.metric("Recall", f"{cm['recall']:.3f}")
        d.metric("F1 score", f"{cm['f1']:.3f}")
        e.metric("ROC-AUC", f"{cm['auc']:.3f}" if np.isfinite(cm["auc"]) else "N/A")
        st.markdown("**Regression – Current property price**")
        a, b, c, d = st.columns(4)
        a.metric("R²", f"{rm['r2']:.3f}")
        b.metric("MAE", f"₹{rm['mae']:.2f} L")
        c.metric("RMSE", f"₹{rm['rmse']:.2f} L")
        d.metric("90th percentile absolute error", f"₹{rm['error90']:.2f} L")
        st.caption("The regression model estimates current price. The five-year figure is a scenario calculation using the selected appreciation rate.")
        imp = st.session_state["importance_df"].copy()
        imp["Feature"] = imp["Feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
        st.plotly_chart(
            px.bar(imp.sort_values("Importance"), x="Importance", y="Feature",
                   orientation="h", title="Top 20 Regression Feature Importances",
                   labels={"Importance": "Feature importance", "Feature": "Input feature"}),
            use_container_width=True
        )
        st.caption("Feature importance indicates how much the fitted Random Forest used each feature; it does not prove causation.")
    else:
        st.info("Click **Train / Load Models** to calculate validation metrics and feature importance. Models are cached after the first training run.")

with tab4:
    st.subheader("🗂️ Filtered Dataset Preview")
    st.write(f"Showing {min(100, len(filtered)):,} of {len(filtered):,} matching properties.")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered properties as CSV", csv, "filtered_housing_properties.csv", "text/csv")

st.divider()
st.caption(
    "Portfolio / educational project only. Results depend on the supplied dataset and selected assumptions. "
    "The dataset may be synthetic or randomized; validate against verified transaction data before making real investment decisions."
)
