import streamlit as st
import numpy as np
import pickle
import pandas as pd


class ClinicalInterface:
    """Interface Streamlit pour la prédiction clinique"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.load_model()

    def load_model(self):
        """Charge le modèle"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.preprocessor = data['preprocessor']

    def run(self):
        """Lance l'interface"""
        st.set_page_config(page_title="Clinical Predictor", page_icon="🏥")

        st.title("🏥 Système de Prédiction Clinique")
        st.markdown("---")

        # Menu latéral
        menu = st.sidebar.selectbox(
            "Menu",
            ["Prédiction Unique", "Prédiction Batch", "Statistiques"]
        )

        if menu == "Prédiction Unique":
            self.single_prediction_page()
        elif menu == "Prédiction Batch":
            self.batch_prediction_page()
        else:
            self.statistics_page()

    def single_prediction_page(self):
        """Page de prédiction unique"""
        st.header("📊 Prédiction pour un patient")

        st.write("Entrez les caractéristiques du patient:")

        # Formulaire dynamique (à adapter selon vos features)
        col1, col2 = st.columns(2)

        with col1:
            feature1 = st.number_input("Feature 1", value=0.0)
            feature2 = st.number_input("Feature 2", value=0.0)
            feature3 = st.number_input("Feature 3", value=0.0)

        with col2:
            feature4 = st.number_input("Feature 4", value=0.0)
            feature5 = st.number_input("Feature 5", value=0.0)

        if st.button("🔍 Analyser", type="primary"):
            # Préparer les données
            features = np.array([[feature1, feature2, feature3, feature4, feature5]])
            features = self.preprocessor.normalize(features, fit=False)

            # Prédiction
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]

            # Affichage des résultats
            st.markdown("---")
            st.subheader("Résultats:")

            if prediction == 1:
                st.error(f"🔴 Diagnostic: **INFECTÉ**")
            else:
                st.success(f"🟢 Diagnostic: **SAIN**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probabilité", f"{probability:.2%}")
            with col2:
                confidence = abs(probability - 0.5) * 2
                st.metric("Confiance", f"{confidence:.2%}")

            # Graphique
            st.progress(float(probability))

    def batch_prediction_page(self):
        """Page de prédiction en batch"""
        st.header("📁 Prédiction Batch")

        uploaded_file = st.file_uploader("Chargez un fichier CSV", type=['csv'])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Aperçu des données:")
            st.dataframe(data.head())

            if st.button("🚀 Lancer les prédictions"):
                X = data.values
                X_normalized = self.preprocessor.normalize(X, fit=False)

                predictions = self.model.predict(X_normalized)
                probabilities = self.model.predict_proba(X_normalized)

                # Ajouter les résultats au dataframe
                data['Prediction'] = predictions
                data['Probabilité'] = probabilities
                data['Diagnostic'] = data['Prediction'].apply(
                    lambda x: "Infecté" if x == 1 else "Sain"
                )

                st.success("Prédictions terminées!")
                st.dataframe(data)

                # Téléchargement des résultats
                csv = data.to_csv(index=False)
                st.download_button(
                    "📥 Télécharger les résultats",
                    csv,
                    "predictions.csv",
                    "text/csv"
                )

    def statistics_page(self):
        """Page de statistiques"""
        st.header("📈 Statistiques du Modèle")

        st.info("Cette section affichera les statistiques une fois les évaluations effectuées.")

        # Exemple de métriques
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", "95.2%")
        with col2:
            st.metric("Precision", "93.8%")
        with col3:
            st.metric("Recall", "96.1%")
        with col4:
            st.metric("F1-Score", "94.9%")