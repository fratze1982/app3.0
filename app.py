import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")

st.title("🎨 KI-Vorhersage für Lackrezepturen")

# 📁 CSV-Datei hochladen
uploaded_file = st.sidebar.file_uploader("CSV-Datei hochladen", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";|,", engine="python")
    
    # 🔁 Komma in Punkt konvertieren (numerische Felder)
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    
    # 🧹 Alle Spaltennamen vereinfachen
    df.columns = [col.strip().replace(" ", "").replace("ß", "ss") for col in df.columns]

    # 🔢 Versuchen, alle numerischen Felder in float zu wandeln
    df = df.apply(pd.to_numeric, errors="ignore")

    # 🎯 Zielspalten definieren (angepasst an deine CSV)
    zielspalten = [
        "KostenGesamtkg", "Viskositaetlowshear", "Viskositaetmidshear", "Brookfield",
        "Glanz20", "Glanz60", "Glanz85", "Kratzschutz"
    ]
    existing_targets = [t for t in zielspalten if t in df.columns]

    if len(existing_targets) == 0:
        st.error("❌ Keine gültigen Zielspalten gefunden.")
        st.stop()

    # 🧪 Eingabe / Ziel trennen
    X = df.drop(columns=existing_targets)
    y = df[existing_targets].apply(pd.to_numeric, errors="coerce")

    # NaN behandeln
    valid_idx = y.dropna().index
    X_clean = X.loc[valid_idx]
    y_clean = y.loc[valid_idx]

    # One-Hot-Encoding für kategoriale Daten
    X_encoded = pd.get_dummies(X_clean)
    
    st.write(f"✅ Verfügbare Trainingsdaten: {X_encoded.shape[0]} Zeilen")

    # 📊 Auswahl Zielgröße für Analyse
    ziel = st.sidebar.selectbox("Zielgröße auswählen", existing_targets)

    # 🧠 Modell trainieren
    modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
    modell.fit(X_encoded, y_clean)

    # 🎛️ Benutzer-Eingabe
    st.sidebar.header("🔧 Eingabewerte")
    user_input = {}
    for col in X.columns:
        if np.issubdtype(df[col].dtype, np.number):
            min_val, max_val = float(df[col].min()), float(df[col].max())
            user_input[col] = st.sidebar.slider(col, min_val, max_val, float(df[col].mean()))
        else:
            user_input[col] = st.sidebar.selectbox(col, df[col].dropna().unique())

    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)
    for col in X_encoded.columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[X_encoded.columns]

    # 🔮 Vorhersage
    vorhersage = modell.predict(input_encoded)[0]
    st.subheader("📈 Vorhersageergebnisse")
    for i, z in enumerate(existing_targets):
        st.metric(z, round(vorhersage[i], 3))

    # 🔍 Partial Dependence Plot
    st.subheader("📊 Einflussfaktoren (Partial Dependence)")
    top_features = st.multiselect("Welche Eingabefelder analysieren?", list(X_encoded.columns), default=list(X_encoded.columns[:3]))

    if len(top_features) > 0:
        fig, ax = plt.subplots(figsize=(12, 6 * len(top_features)))
        PartialDependenceDisplay.from_estimator(modell, X_encoded, features=top_features, target=existing_targets.index(ziel), ax=ax)
        st.pyplot(fig)

    # 💬 Freitextfrage (nur lokal mit Regeln)
    st.subheader("💬 Fragen stellen (Regelbasierter Assistent)")
    frage = st.text_input("Was möchtest du wissen?")
    if frage:
        antwort = ""
        if "glanz" in frage.lower():
            antwort = "Der Glanzwert sinkt bei höherem Einsatz von Sylysia (Mattierungsmittel)."
        elif "kosten" in frage.lower():
            antwort = "Die Gesamtkosten steigen i. d. R. mit mehr Lackslurry oder teuren Additiven."
        elif "viskosität" in frage.lower():
            antwort = "Die Viskosität wird u. a. von Acrysol-Additiven beeinflusst."
        else:
            antwort = "Diese Frage kann aktuell nur regelbasiert beantwortet werden."
        st.success(antwort)
else:
    st.info("⬅️ Bitte lade eine CSV-Datei hoch.")
