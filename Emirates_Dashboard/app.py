import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Emirates Airways Reviews Dashboard", layout="wide")

DEFAULT_PATH = "Emirates Airways Reviews.csv"

NUMERIC_FEATURES = [
    "Seating Comfort", "Staff Service", "Food Quality",
    "Entertainment", "WiFi", "Ground Service", "Value for Money",
]
CATEGORICAL_FEATURES = ["Travel Type", "Travel Class", "Recommended", "Country"]
TARGET = "Overall Rating"
TOP_COUNTRIES = 15


@st.cache_data
def load_data(file_bytes, file_name):
    if file_bytes is not None:
        from io import BytesIO
        df = pd.read_csv(BytesIO(file_bytes))
    else:
        df = pd.read_csv(file_name)

    # Group rare countries into "Other" so the model/encoder stays compact
    top = df["Country"].value_counts().nlargest(TOP_COUNTRIES).index
    df["Country"] = df["Country"].where(df["Country"].isin(top), "Other")
    return df


@st.cache_resource
def train_model(df):
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    n = len(y_test)
    p = pipeline.named_steps["preprocess"].transform(X_test).shape[1]
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return pipeline, {"r2": r2, "adj_r2": adj_r2, "mae": mae, "rmse": rmse, "n_test": n}


# --- Sidebar: data upload ---
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload an updated reviews CSV", type="csv")
file_bytes = uploaded.getvalue() if uploaded is not None else None
file_name = uploaded.name if uploaded is not None else DEFAULT_PATH

df = load_data(file_bytes, file_name)
st.sidebar.caption(f"Loaded {len(df):,} reviews from {'uploaded file' if uploaded else 'default dataset'}.")

# --- Sidebar: filters ---
st.sidebar.header("Filters")
travel_types = st.sidebar.multiselect("Travel Type", sorted(df["Travel Type"].dropna().unique()))
travel_classes = st.sidebar.multiselect("Travel Class", sorted(df["Travel Class"].dropna().unique()))
countries = st.sidebar.multiselect("Country", sorted(df["Country"].dropna().unique()))
recommended = st.sidebar.multiselect("Recommended", sorted(df["Recommended"].dropna().unique()))
rating_range = st.sidebar.slider("Overall Rating range", 1, 10, (1, 10))

filtered = df.copy()
if travel_types:
    filtered = filtered[filtered["Travel Type"].isin(travel_types)]
if travel_classes:
    filtered = filtered[filtered["Travel Class"].isin(travel_classes)]
if countries:
    filtered = filtered[filtered["Country"].isin(countries)]
if recommended:
    filtered = filtered[filtered["Recommended"].isin(recommended)]
filtered = filtered[(filtered[TARGET] >= rating_range[0]) & (filtered[TARGET] <= rating_range[1])]

st.title("✈️ Emirates Airways Reviews Dashboard")

# --- KPIs ---
st.subheader("Key Metrics")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Reviews", f"{len(filtered):,}")
k2.metric("Avg Overall Rating", f"{filtered[TARGET].mean():.2f}" if len(filtered) else "—")
pct_recommended = (filtered["Recommended"].eq("yes").mean() * 100) if len(filtered) else 0
k3.metric("% Recommended", f"{pct_recommended:.1f}%")
k4.metric("Avg Value for Money", f"{filtered['Value for Money'].mean():.2f}" if len(filtered) else "—")
k5.metric("Top Travel Class", filtered["Travel Class"].mode().iloc[0] if len(filtered) else "—")

st.divider()

# --- Charts ---
st.subheader("Charts")
sns.set_style("whitegrid")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Overall Rating distribution**")
    fig, ax = plt.subplots()
    sns.histplot(filtered[TARGET], bins=10, ax=ax)
    ax.set_xlabel("Overall Rating")
    st.pyplot(fig)

with c2:
    st.markdown("**Average rating by Travel Class**")
    fig, ax = plt.subplots()
    order = filtered.groupby("Travel Class")[TARGET].mean().sort_values(ascending=False)
    sns.barplot(x=order.index, y=order.values, ax=ax)
    ax.set_ylabel("Avg Overall Rating")
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Reviews by Country (top 10)**")
    fig, ax = plt.subplots()
    counts = filtered["Country"].value_counts().head(10)
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_xlabel("Number of reviews")
    st.pyplot(fig)

with c4:
    st.markdown("**Recommended share**")
    fig, ax = plt.subplots()
    rec_counts = filtered["Recommended"].value_counts()
    ax.pie(rec_counts.values, labels=rec_counts.index, autopct="%1.0f%%")
    st.pyplot(fig)

st.markdown("**Correlation between sub-ratings and Overall Rating**")
fig, ax = plt.subplots(figsize=(8, 5))
corr = filtered[NUMERIC_FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.divider()

# --- Model ---
st.subheader("Overall Rating Prediction Model")
st.caption(
    "A Linear Regression model trained on the numeric sub-ratings and key categorical fields "
    "(Travel Type, Travel Class, Recommended, Country). Free-text fields (Title, Review, dates, Route, Status) "
    "are excluded. Missing sub-ratings are median-imputed before training."
)

pipeline, metrics = train_model(df)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("R²", f"{metrics['r2']:.3f}")
m2.metric("Adjusted R²", f"{metrics['adj_r2']:.3f}")
m3.metric("MAE", f"{metrics['mae']:.2f}")
m4.metric("RMSE", f"{metrics['rmse']:.2f}")
m5.metric("Test set size", f"{metrics['n_test']:,} (20%)")

st.divider()

# --- Prediction form ---
st.subheader("Try the model: predict an Overall Rating")
with st.form("prediction_form"):
    col_a, col_b = st.columns(2)
    with col_a:
        seating = st.slider("Seating Comfort", 1, 5, 3)
        staff = st.slider("Staff Service", 1, 5, 3)
        food = st.slider("Food Quality", 1, 5, 3)
        entertainment = st.slider("Entertainment", 1, 5, 3)
    with col_b:
        wifi = st.slider("WiFi", 1, 5, 3)
        ground = st.slider("Ground Service", 1, 5, 3)
        value = st.slider("Value for Money", 1, 5, 3)
        travel_type = st.selectbox("Travel Type", sorted(df["Travel Type"].dropna().unique()))
    col_c, col_d = st.columns(2)
    with col_c:
        travel_class = st.selectbox("Travel Class", sorted(df["Travel Class"].dropna().unique()))
        country = st.selectbox("Country", sorted(df["Country"].dropna().unique()))
    with col_d:
        recommended_input = st.selectbox("Recommended", sorted(df["Recommended"].dropna().unique()))

    submitted = st.form_submit_button("Predict Overall Rating")

if submitted:
    input_row = pd.DataFrame([{
        "Seating Comfort": seating,
        "Staff Service": staff,
        "Food Quality": food,
        "Entertainment": entertainment,
        "WiFi": wifi,
        "Ground Service": ground,
        "Value for Money": value,
        "Travel Type": travel_type,
        "Travel Class": travel_class,
        "Recommended": recommended_input,
        "Country": country,
    }])
    prediction = pipeline.predict(input_row)[0]
    prediction_clamped = float(np.clip(prediction, 1, 10))
    st.success(f"### Predicted Overall Rating: {prediction_clamped:.1f} / 10")
