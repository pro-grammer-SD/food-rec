import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Smart Food Recommender", page_icon="🥗", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("food.csv").fillna(0)

df = load_data()

num_cols = [
    "Data.Alpha Carotene","Data.Beta Carotene","Data.Beta Cryptoxanthin",
    "Data.Carbohydrate","Data.Cholesterol","Data.Choline","Data.Fiber",
    "Data.Lutein and Zeaxanthin","Data.Lycopene","Data.Niacin","Data.Protein",
    "Data.Retinol","Data.Riboflavin","Data.Selenium","Data.Sugar Total",
    "Data.Thiamin","Data.Water","Data.Fat.Monosaturated Fat",
    "Data.Fat.Polysaturated Fat","Data.Fat.Saturated Fat",
    "Data.Fat.Total Lipid","Data.Major Minerals.Calcium",
    "Data.Major Minerals.Copper","Data.Major Minerals.Iron",
    "Data.Major Minerals.Magnesium","Data.Major Minerals.Phosphorus",
    "Data.Major Minerals.Potassium","Data.Major Minerals.Sodium",
    "Data.Major Minerals.Zinc","Data.Vitamins.Vitamin A - RAE",
    "Data.Vitamins.Vitamin B12","Data.Vitamins.Vitamin B6",
    "Data.Vitamins.Vitamin C","Data.Vitamins.Vitamin E",
    "Data.Vitamins.Vitamin K"
]

@st.cache_resource
def build_model(data):
    preprocess = ColumnTransformer([
        ("text", TfidfVectorizer(stop_words="english", max_features=600), "Description"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Category"]),
        ("id", StandardScaler(), ["Nutrient Data Bank Number"]),
        ("num", StandardScaler(), num_cols)
    ])
    X = preprocess.fit_transform(data)
    model = NearestNeighbors(n_neighbors=10, metric="cosine")
    model.fit(X)
    return preprocess, model

preprocess, model = build_model(df)

st.markdown(
    """
    <style>
    .card {
        padding: 1.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #ffffff, #f2f4f8);
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🥗 Smart Food Recommender")
st.caption("Pick your goal. Describe your vibe. Get food that actually matches.")

col1, col2, col3 = st.columns(3)

with col1:
    category = st.selectbox(
        "Food Category",
        [""] + sorted(df["Category"].unique().tolist())
    )

with col2:
    goal = st.selectbox(
        "Main Goal",
        [
            "Build muscle / High protein",
            "Lose weight / Low fat & sugar",
            "Heart healthy / Low sodium & fat",
            "Vitamin rich / Immunity boost"
        ]
    )

with col3:
    diet = st.selectbox(
        "Diet Preference",
        [
            "No preference",
            "Low sugar",
            "Low fat",
            "Low sodium"
        ]
    )

description = st.text_input(
    "Describe your ideal food",
    placeholder="light, healthy, high protein, easy to digest"
)

run = st.button("Find Food 🔥", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = []

if run:
    query = df.mean(numeric_only=True)
    query["Description"] = description if description else "healthy food"
    query["Category"] = category if category else df["Category"].mode()[0]
    query["Nutrient Data Bank Number"] = df["Nutrient Data Bank Number"].median()

    if goal == "Build muscle / High protein":
        query["Data.Protein"] = df["Data.Protein"].quantile(0.9)
    if goal == "Lose weight / Low fat & sugar":
        query["Data.Fat.Total Lipid"] = df["Data.Fat.Total Lipid"].quantile(0.1)
        query["Data.Sugar Total"] = df["Data.Sugar Total"].quantile(0.1)
    if goal == "Heart healthy / Low sodium & fat":
        query["Data.Major Minerals.Sodium"] = df["Data.Major Minerals.Sodium"].quantile(0.1)
        query["Data.Fat.Total Lipid"] = df["Data.Fat.Total Lipid"].quantile(0.2)
    if goal == "Vitamin rich / Immunity boost":
        query["Data.Vitamins.Vitamin C"] = df["Data.Vitamins.Vitamin C"].quantile(0.8)
        query["Data.Vitamins.Vitamin A - RAE"] = df["Data.Vitamins.Vitamin A - RAE"].quantile(0.8)

    if diet == "Low sugar":
        query["Data.Sugar Total"] = df["Data.Sugar Total"].quantile(0.1)
    if diet == "Low fat":
        query["Data.Fat.Total Lipid"] = df["Data.Fat.Total Lipid"].quantile(0.1)
    if diet == "Low sodium":
        query["Data.Major Minerals.Sodium"] = df["Data.Major Minerals.Sodium"].quantile(0.1)

    query_df = pd.DataFrame([query])
    X_query = preprocess.transform(query_df)
    _, indices = model.kneighbors(X_query)

    st.session_state.results.append(indices[0])

if st.session_state.results:
    st.subheader("🔥 Your Recommendations")

    cols = st.columns(2)
    shown = set()

    for batch in st.session_state.results:
        for idx, i in enumerate(batch):
            if i in shown:
                continue
            shown.add(i)
            row = df.iloc[i]
            with cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="card">
                        <h4>{row['Description']}</h4>
                        <p><b>Category:</b> {row['Category']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
