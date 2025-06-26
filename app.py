import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt

# -------------------------------
# CSV-Daten laden
# -------------------------------
st.set_page_config(layout="wide")
st.title("🎨 KI-Vorhersage für Lackrezepturen")
csv_file = "rezeptdaten.csv"
df = pd.read_csv(csv_file, encoding="utf-8", sep=";")

# -------------------------------
# Zielgrößen definieren (anpassbar)
# -------------------------------
all_targets = [
    "KostenGesamtkg", "Viskositätlowshear", "Viskositätmidshear", "Brookfield",
    "Glanz20", "Glanz60", "Glanz85", "Kratzschutz"
]
existing_targets = [col for col in all_targets if col in df.columns]

# -------------------------------
# Eingabe- und Ausgabedaten trennen
# -------------------------------
df_clean = df.dropna(subset=existing_targets).copy()
y_clean = df_clean[existing_targets].apply(pd.to_numeric, errors='coerce')
X = df_clean.drop(columns=existing_targets)

# Kategorische und numerische Spalten identifizieren
kategorisch = X.select_dtypes(include="object").columns.tolist()
numerisch = X.select_dtypes(exclude="object").columns.tolist()

# One-Hot-Encoding
X_encoded = pd.get_dummies(X)
X_encoded_clean = X_encoded.loc[y_clean.dropna().index]
y_clean = y_clean.dropna()

# Abbruch wenn Daten leer
if len(X_encoded_clean) == 0 or len(y_clean) == 0:
    st.error("❌ Keine gültigen Trainingsdaten verfügbar. Bitte überprüfe Zielspalten und Werte.")
    st.stop()

# -------------------------------
# Modell trainieren
# -------------------------------
modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
modell.fit(X_encoded_clean, y_clean)

# -------------------------------
# Zielgrößen-Auswahl für Analyse
# -------------------------------
st.sidebar.subheader("🎯 Zielgrößen auswählen")
selected_targets = st.sidebar.multiselect("Zielgrößen", existing_targets, default=existing_targets[:2])

# -------------------------------
# Eingabeformular für Nutzer
# -------------------------------
st.sidebar.subheader("🧪 Neue Rezeptur eingeben")
user_input = {}
for col in numerisch:
    min_val, max_val, mean_val = float(df[col].min()), float(df[col].max()), float(df[col].mean())
    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

for col in kategorisch:
    user_input[col] = st.sidebar.selectbox(col, sorted(df[col].dropna().unique()))

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Fehlende Spalten auffüllen
for col in X_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_encoded.columns]

# -------------------------------
# Vorhersage für Eingabe anzeigen
# -------------------------------
prediction = modell.predict(input_encoded)[0]
st.subheader("🔮 Vorhergesagte Eigenschaften")
cols = st.columns(len(selected_targets))
for i, ziel in enumerate(selected_targets):
    if ziel in existing_targets:
        val = prediction[existing_targets.index(ziel)]
        cols[i].metric(label=ziel, value=round(val, 2))

# -------------------------------
# 📊 Partial Dependence Plot
# -------------------------------
st.markdown("---")
st.subheader("📈 Abhängigkeitsanalyse einzelner Variablen")

with st.expander("🧪 Variable analysieren"):
    feature = st.selectbox("Feature für Analyse auswählen", X_encoded_clean.columns)
    target_name = st.selectbox("Zielgröße für PDP", existing_targets)
    target_idx = existing_targets.index(target_name)
    feat_idx = X_encoded_clean.columns.get_loc(feature)

    try:
        pdp_result = partial_dependence(modell, X_encoded_clean, features=[feat_idx], target=target_idx)
        values = pdp_result["values"][0]
        ave_pred = pdp_result["average"][0]

        fig, ax = plt.subplots()
        ax.plot(values, ave_pred, marker="o")
        ax.set_xlabel(feature)
        ax.set_ylabel(f"{target_name} (mittlere Vorhersage)")
        ax.set_title(f"Partial Dependence: {target_name} vs {feature}")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Keine PDP möglich: {e}")

# -------------------------------
# 🗨️ Regelbasiertes Fragefeld
# -------------------------------
st.markdown("---")
st.subheader("🧠 Regelbasierte Analyse")
frage = st.text_input("Stelle eine Frage zur Rezeptur")

if frage:
    antwort = ""
    frage_lower = frage.lower()
    if "lackslurry" in frage_lower and "kosten" in frage_lower:
        antwort = "💡 Höherer Lackslurry-Anteil erhöht meist die Rohstoffkosten pro kg."
    elif "sylysia" in frage_lower:
        antwort = "💡 Sylysia wird als Mattierungsmittel verwendet – hoher Anteil senkt den Glanz und kann Viskosität erhöhen."
    elif "glanz" in frage_lower and "viskosität" in frage_lower:
        antwort = "🔁 Zwischen Glanz und Viskosität gibt es häufig einen Zielkonflikt – starke Mattierung erhöht oft die Viskosität."
    else:
        antwort = "🤔 Die Frage konnte nicht automatisch beantwortet werden. Bitte spezifischer formulieren."
    st.info(antwort)

# -------------------------------
# Export
# -------------------------------
st.markdown("---")
st.download_button("📥 Eingabe & Vorhersage als CSV", data=input_df.assign(**{ziel: prediction[i] for i, ziel in enumerate(existing_targets)}).to_csv(index=False), file_name="vorhersage.csv", mime="text/csv")
