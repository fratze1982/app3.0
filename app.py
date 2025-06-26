import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# ------------------------------
# Datei laden und vorbereiten
# ------------------------------
st.title("🎨 KI-Vorhersage für Lackrezepturen")

uploaded_file = st.file_uploader("Lade deine CSV-Datei hoch", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip().str.replace(" ", "").str.replace("\t", "")

    # Zielspalten
    targets = [
        "Glanz20", "Glanz60", "Glanz85",
        "Viskositätlowshear", "Viskositätmidshear", "Brookfield",
        "KostenGesamtkg", "Kratzschutz"
    ]
    existing_targets = [t for t in targets if t in df.columns]

    X = df.drop(columns=existing_targets, errors="ignore")
    y = df[existing_targets]

    # NaNs bereinigen
    df_clean = pd.concat([X, y], axis=1).dropna()
    X_clean = df_clean.drop(columns=existing_targets)
    y_clean = df_clean[existing_targets].astype(float)

    # One-Hot-Encoding
    X_encoded = pd.get_dummies(X_clean)
    X_encoded_clean = X_encoded.copy()

    # Modell trainieren
    modell = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, random_state=42))
    modell.fit(X_encoded_clean, y_clean)

    # Numerisch/kategorisch trennen
    kategorisch = X_clean.select_dtypes(include="object").columns.tolist()
    numerisch = X_clean.select_dtypes(exclude="object").columns.tolist()

    # Sidebar Eingabe
    st.sidebar.header("🔧 Eingabewerte anpassen")
    user_input = {}
    for col in numerisch:
        min_val, max_val, mean_val = float(df[col].min()), float(df[col].max()), float(df[col].mean())
        user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
    for col in kategorisch:
        user_input[col] = st.sidebar.selectbox(col, sorted(df[col].dropna().unique()))

    # Eingabe vorbereiten
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)
    for col in X_encoded_clean.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[X_encoded_clean.columns]

    prediction = modell.predict(input_encoded)[0]

    st.subheader("🔮 Vorhergesagte Eigenschaften")
    for i, ziel in enumerate(existing_targets):
        st.metric(label=ziel, value=round(prediction[i], 2))

    # Exportbereich
    export_data = input_df.copy()
    for i, ziel in enumerate(existing_targets):
        export_data[f"Vorhersage: {ziel}"] = prediction[i]
    csv = export_data.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Einzelvorhersage als CSV", csv, file_name="rezeptur_vorhersage.csv", mime="text/csv")

    # ----------------------------------------
    # Mehrere Varianten eingeben
    # ----------------------------------------
    st.markdown("---")
    st.subheader("📊 Vergleich mehrerer Rezeptvarianten")
    sample_variants = pd.DataFrame([user_input]*3)
    sample_variants.iloc[1, sample_variants.columns.get_loc(numerisch[0])] *= 0.9
    sample_variants.iloc[2, sample_variants.columns.get_loc(numerisch[0])] *= 1.1

    edited_df = st.data_editor(sample_variants, use_container_width=True, num_rows="dynamic")

    if st.button("🚀 Vorhersage starten für alle Varianten"):
        encoded_variants = pd.get_dummies(edited_df)
        for col in X_encoded_clean.columns:
            if col not in encoded_variants.columns:
                encoded_variants[col] = 0
        encoded_variants = encoded_variants[X_encoded_clean.columns]

        predictions = modell.predict(encoded_variants)
        results_df = edited_df.copy()
        for i, ziel in enumerate(existing_targets):
            results_df[f"Vorhersage: {ziel}"] = predictions[:, i]

        st.dataframe(results_df, use_container_width=True)

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Alle Vorhersagen als CSV", data=csv, file_name="rezeptur_vergleich.csv", mime="text/csv")

        # ----------------------------------------
        # Visualisierung
        # ----------------------------------------
        st.markdown("---")
        st.subheader("📈 Visualisierung der Abhängigkeiten")
        x_axis = st.selectbox("X-Achse", results_df.columns)
        y_axis = st.selectbox("Y-Achse", results_df.columns, index=results_df.columns.get_loc("Vorhersage: Glanz60") if "Vorhersage: Glanz60" in results_df.columns else 0)

        fig, ax = plt.subplots()
        ax.scatter(results_df[x_axis], results_df[y_axis], color="steelblue")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{y_axis} vs. {x_axis}")
        st.pyplot(fig)

        # ----------------------------------------
        # Fragenfeld (lokale Regel-Antworten)
        # ----------------------------------------
        st.markdown("---")
        st.subheader("🧠 Fragen zur Rezeptur stellen")
        user_question = st.text_input("Was möchtest du über die Rezeptur wissen?")

        if user_question:
            antwort = f"Deine Frage war: *{user_question}*\n\n📊 Basierend auf den aktuellen Daten:\n"
            if "Sylysia256" in results_df.columns and "Vorhersage: Glanz60" in results_df.columns:
                corr = results_df["Sylysia256"].corr(results_df["Vorhersage: Glanz60"])
                richtung = "steigt" if corr > 0 else "sinkt"
                antwort += f"- Wenn du mehr **Sylysia256** einsetzt, {richtung} der **Glanz60** tendenziell (Korrelation: {corr:.2f}).\n"

            kosten_cols = [col for col in results_df.columns if "Kosten" in col]
            if kosten_cols:
                teuerste = results_df[kosten_cols].mean().idxmax()
                antwort += f"- Die größten mittleren Kosten entstehen durch **{teuerste}**.\n"

            st.markdown(antwort)
else:
    st.info("⬆️ Bitte lade zuerst eine CSV-Datei hoch.")
