import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("food.csv").fillna(0)

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

preprocess = ColumnTransformer([
    ("text", TfidfVectorizer(stop_words="english", max_features=600), "Description"),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Category"]),
    ("id", StandardScaler(), ["Nutrient Data Bank Number"]),
    ("num", StandardScaler(), num_cols)
])

X = preprocess.fit_transform(df)

model = NearestNeighbors(n_neighbors=10, metric="cosine")
model.fit(X)

print("\n🍽️  WELCOME TO THE SMART FOOD RECOMMENDER\n")
print("Answer a few simple questions and I’ll suggest foods that match your goals.\n")

category = input("1️⃣ What kind of food are you looking for?\n   (Examples: Milk, Vegetables, Fruits, Meat, Snacks)\n   Press Enter to skip: ").strip()

goal = input("\n2️⃣ What is your main goal?\n   a) Build muscle / High protein\n   b) Lose weight / Low fat & sugar\n   c) Heart healthy / Low sodium & fat\n   d) Vitamin rich / Immunity boost\n   Type a, b, c, or d: ").strip().lower()

diet = input("\n3️⃣ Any specific dietary concern?\n   a) Low sugar\n   b) Low fat\n   c) Low sodium\n   d) No preference\n   Type a, b, c, or d: ").strip().lower()

description = input("\n4️⃣ Describe the food in your own words\n   (Example: light, healthy, high protein, easy to digest): ").strip()

query = df.mean(numeric_only=True)
query["Description"] = description
query["Category"] = category if category else df["Category"].mode()[0]
query["Nutrient Data Bank Number"] = df["Nutrient Data Bank Number"].median()

if goal == "a":
    query["Data.Protein"] = df["Data.Protein"].quantile(0.9)
if goal == "b":
    query["Data.Fat.Total Lipid"] = df["Data.Fat.Total Lipid"].quantile(0.1)
    query["Data.Sugar Total"] = df["Data.Sugar Total"].quantile(0.1)
if goal == "c":
    query["Data.Major Minerals.Sodium"] = df["Data.Major Minerals.Sodium"].quantile(0.1)
    query["Data.Fat.Total Lipid"] = df["Data.Fat.Total Lipid"].quantile(0.2)
if goal == "d":
    query["Data.Vitamins.Vitamin C"] = df["Data.Vitamins.Vitamin C"].quantile(0.8)
    query["Data.Vitamins.Vitamin A - RAE"] = df["Data.Vitamins.Vitamin A - RAE"].quantile(0.8)

if diet == "a":
    query["Data.Sugar Total"] = df["Data.Sugar Total"].quantile(0.1)
if diet == "b":
    query["Data.Fat.Total Lipid"] = df["Data.Fat.Total Lipid"].quantile(0.1)
if diet == "c":
    query["Data.Major Minerals.Sodium"] = df["Data.Major Minerals.Sodium"].quantile(0.1)

query_df = pd.DataFrame([query])

X_query = preprocess.transform(query_df)
distances, indices = model.kneighbors(X_query)

print("\n🔥 TOP FOOD RECOMMENDATIONS FOR YOU:\n")

for i in indices[0]:
    row = df.iloc[i]
    print(f"• {row['Category']} — {row['Description']}")
    