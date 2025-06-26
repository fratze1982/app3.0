import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="KI-Vorhersage für Lackrezepturen", layout="wide")
st.title("🎨 KI-Vorhersage für Lackrezepturen")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("📁 CSV-Datei hochladen", type=["csv"])

if uploaded_file is None:
    st.warning("Bitte lade eine CSV-Datei hoch.")
    st.stop()

# --- Daten einlesen ---
try:
    df = pd.read_csv(uploaded_file, sep=",", decimal=",")
except Exception as e:
    st.error(f"Fehler beim Einlesen der Datei: {e}")
    st.stop()

# --- Zielspalten definieren ---
alle_möglichen_targets = [
    "KostenGesamtkg", "Viskositätlowshear", "Viskositätmidshear",
    "Brookfield", "Glanz20", "Glanz60", "Glanz85", "Kratzschutz"
]
existing_targets = [t for t in alle_möglichen_targets if t in df.columns]

if not existing_targets:
    st.error("❌ Keine gültigen Zielspalten gefunden.")
    st.stop()

# --- Zielgrößen-Auswahl (Mehrfachauswahl möglich) ---
zielspalten = st.multiselect("🎯 Zielgrößen auswählen", options=existing_targets, default=["Brookfield"])
if not zielspalten:
    st.warning("Bitte mindestens eine Zielgröße auswählen.")
    st.stop()

# --- Aufbereitung ---
X = df.drop(columns=[col for col in zielspalten if col in df.columns], errors="ignore")
y = df[zielspalten].copy()

# Typen prüfen
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding
X_encoded = pd.get_dummies(X)

# NaNs bereinigen
df_encoded = X_encoded.copy()
df_encoded[y.columns] = y
df_encoded = df_encoded.dropna()

X_encoded_clean = df_encoded[X_encoded.columns]
y_clean = df_encoded[y.columns]

if X_encoded_clean.empty or y_clean.empty:
    st.error("❌ Keine gültigen Trainingsdaten verfügbar. Bitte überprüfe Zielspalten und Werte.")
    st.stop()

# --- Modelltraining ---
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_encoded_clean, y_clean)

# --- Eingabeformular ---
st.sidebar.header("🔧 Eingabewerte anpassen")
user_input = {}
for col in numerisch:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

for col in kategorisch:
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.sidebar.selectbox(col, options)

# --- Benutzer-Eingabe vorbereiten ---
input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Fehlende Spalten auffüllen
for col in X_encoded_clean.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_encoded_clean.columns]

# --- Vorhersage ---
prediction = modell.predict(input_encoded)[0]

st.subheader("🔮 Vorhergesagte Eigenschaften")
for i, ziel in enumerate(zielspalten):
    st.metric(label=ziel, value=round(prediction[i], 2))

# --- Analyse: Partial Dependence Plot ---
st.subheader("📊 Abhängigkeitsanalyse (Partial Dependence)")

feature_options = X_encoded_clean.columns.tolist()
selected_feature = st.selectbox("📌 Feature auswählen", feature_options)
selected_targets = st.multiselect("📈 Zielgrößen für Analyse", zielspalten, default=zielspalten[:1])

if selected_feature and selected_targets:
    for ziel in selected_targets:
        try:
            target_index = zielspalten.index(ziel)
            pdp_result = partial_dependence(modell, X_encoded_clean, features=[selected_feature], target=target_index)
            x_vals = pdp_result["values"][0]
            y_vals = pdp_result["average"][0]

            fig, ax = plt.subplots()
            sns.lineplot(x=x_vals, y=y_vals, ax=ax)
            ax.set_title(f"Partial Dependence: {ziel} vs. {selected_feature}")
            ax.set_xlabel(selected_feature)
            ax.set_ylabel(ziel)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"PDP konnte für {ziel} nicht berechnet werden: {e}")

# --- Fragen-Feld für lokale Regeln ---
st.subheader("💬 Fragen zu Wechselwirkungen (lokale Regeln)")
frage = st.text_input("🧠 Formuliere eine Frage (z. B. 'Wie wirkt sich Sylysia256 auf Glanz20 aus?')")

if frage:
    st.info("ℹ️ Diese Funktion ist aktuell lokal regelbasiert. Beispielausgaben folgen.")
    if "sylysia" in frage.lower() and "glanz" in frage.lower():
        st.success("📌 Mehr Sylysia256 → tendenziell geringerer Glanz (Mattierung).")
    elif "lackslurry" in frage.lower() and "kosten" in frage.lower():
        st.success("📌 Höherer Lackslurry-Anteil → höhere KostenGesamtkg.")
    else:
        st.warning("🔍 Noch keine Regel für diese Frage definiert.")
