import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import seaborn as sns

# CSV-Datei laden
df = pd.read_csv("/mnt/data/rezeptdaten.csv", sep=",", decimal=",")

# Zielgrößen festlegen
targets = [
    "KostenGesamtkg", "Viskositätlowshear", "Viskositätmidshear",
    "Brookfield", "Glanz20", "Glanz60", "Glanz85", "Kratzschutz"
]

# Verfügbare Spalten prüfen
available_targets = [t for t in targets if t in df.columns]
if not available_targets:
    st.error("❌ Keine gültigen Zielspalten in den Daten gefunden.")
    st.stop()

# Eingabe- und Zielgrößen trennen
X = df.drop(columns=available_targets)
y = df[available_targets]

# NaNs entfernen
X_clean = X.dropna()
y_clean = y.loc[X_clean.index]

# Modell trainieren
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_clean, y_clean)

# Streamlit UI
st.title("🎨 KI-Vorhersage und Analyse von Lackrezepturen")
st.markdown("Wähle Zielgrößen zur Analyse, passe Eingaben an und analysiere Zusammenhänge.")

# Zielgrößen-Auswahl
selected_targets = st.multiselect("🎯 Zielgrößen auswählen", available_targets, default=["Brookfield"])

# Eingabewerte anpassen
st.sidebar.header("🔧 Rezeptureingaben")
user_input = {}
for col in X.columns:
    min_val, max_val = float(X[col].min()), float(X[col].max())
    mean_val = float(X[col].mean())
    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

input_df = pd.DataFrame([user_input])

# Vorhersage
prediction = modell.predict(input_df)[0]

# Vorhersage anzeigen
st.subheader("📈 Vorhergesagte Zielgrößen")
for i, ziel in enumerate(available_targets):
    if ziel in selected_targets:
        st.metric(label=ziel, value=round(prediction[i], 2))

# Analyse: Partial Dependence Plots
st.subheader("🔍 Einfluss einzelner Merkmale auf Zielgrößen")
for ziel in selected_targets:
    ziel_idx = available_targets.index(ziel)
    for feature in X.columns:
        st.markdown(f"**Einfluss von {feature} auf {ziel}:**")
        fig, ax = plt.subplots()
        pd_result = partial_dependence(modell, X_clean, features=[X.columns.get_loc(feature)], target=ziel_idx)
        ax.plot(pd_result['values'][0], pd_result['average'][0])
        ax.set_xlabel(feature)
        ax.set_ylabel(ziel)
        ax.grid(True)
        st.pyplot(fig)

# Optional: Freitext-Erklärungen (lokal regelbasiert)
st.subheader("🧠 Erklärung durch einfache Regeln")
rules = []
if user_input.get("Sylysia256", 0) > X["Sylysia256"].mean():
    rules.append("🔻 Hoher Einsatz von Mattierungsmittel (Sylysia256) kann zu geringerem Glanz führen.")
if user_input.get("Lackslurry", 0) > X["Lackslurry"].mean():
    rules.append("💰 Mehr Lackslurry kann die Kosten steigern.")
if user_input.get("AcrysolRM2020E", 0) > X["AcrysolRM2020E"].mean():
    rules.append("💧 Höherer Einsatz von Rheologieadditiven beeinflusst Viskosität.")

if rules:
    for r in rules:
        st.info(r)
else:
    st.write("Keine speziellen Regeln für die aktuelle Eingabe erkannt.")
