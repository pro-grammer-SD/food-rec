# 🥗 Smart Food Recommender

A machine‑learning powered food recommendation system that works both as a CLI tool and a modern Streamlit web app. It analyzes nutritional data, food categories, and natural‑language descriptions to suggest foods aligned with user goals.

This project is designed to be:

- Practical

- Fast

- Explainable

- Deployable


No gimmicks. **Just clean ML + usable UX.**


---

📁 Project Structure

```bash
├── food.csv
├── main.py
├── app.py
├── requirements.txt
└── README.md
```

## food.csv

The dataset containing food items, categories, textual descriptions, and detailed nutritional information.


---

## 🖥️ CLI Application (main.py)

The CLI version is built for terminal users who prefer a guided, question‑based experience.

### What it does

- Asks the user a series of questions

- Builds a synthetic “ideal food” profile

- Uses TF‑IDF + nutrient scaling

- Finds the closest foods using cosine similarity

- Prints ranked recommendations directly in the terminal


### How to run

```bash
python main.py
```

### User Flow

1. Choose a food category (optional)


2. Select a health goal


3. Select dietary constraints


4. Describe the food in natural language


5. Receive top matching food suggestions

## Best for

- Quick experiments

- Terminal lovers

- Debugging model logic

- Lightweight environments

---

## 🌐 GUI Application (app.py – Streamlit)

The Streamlit app is a full‑fledged graphical interface built for usability and exploration.

### Key UX Features

All controls on the main screen (no sidebar)

Persistent UI (results do not reset inputs)

Search history accumulates like a feed

Card‑based layout for readability

Responsive, modern design


### How to run

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

### App Flow

1. Select food category


2. Select health goal


3. Select diet preference


4. Add a free‑text description


5. Click Find Food 🔥


6. Scroll through stacked recommendation cards
7. 

Each click adds new recommendations instead of replacing previous ones.

### Best for

- End users

- Demos

- Deployment

- Product‑style usage

---

## 🧠 Machine Learning Pipeline

- The same ML pipeline is shared by both CLI and GUI.

- Feature Engineering

- Text: TF‑IDF Vectorization on food descriptions

- Category: One‑Hot Encoding

- Nutrients: Standard Scaling on 30+ nutritional values

- ID: Scaled numeric identifier

## Model

- Algorithm: Nearest Neighbors

- Metric: Cosine similarity

- Purpose: Content‑based recommendation

## Why this works

- No cold‑start problem

- Interpretable

- Fast on CPU

- No training loop needed

---

## 📦 Installation for local deployments

Create a virtual environment (recommended), then install dependencies.

`pip install -r requirements.txt`

## Requirements

- Python 3.9+

- Streamlit

- Pandas

- NumPy

- Scikit‑learn

---

## 🧪 Notes

- Both main.py and app.py use the same data logic

- No external APIs required

- Fully offline capable



---

## 🧠 Philosophy

- CLI for control. GUI for comfort. ML for brains.

- Use what fits your flow.


---

## 🏁 TL;DR

Want terminal? Run main.py

Want UI? Run app.py

Same brain, different faces


Happy recommending 🥗🔥
