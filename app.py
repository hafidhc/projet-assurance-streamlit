import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# ==============================================================================
# PARTIE 1 : LE CŒUR DU DATA SCIENTIST (ENTRAÎNEMENT DU MODÈLE)
# ==============================================================================

MODEL_FILE = 'charge_sinistre_model.pkl'

# Cette fonction est mise en cache. Elle ne s'exécutera qu'une seule fois.
@st.cache_resource
def train_model():
    """
    Cette fonction simule la création d'un jeu de données, entraîne un modèle
    de Machine Learning, et le sauvegarde pour une utilisation future.
    """
    # Création de données simulées mais réalistes pour l'assurance auto
    size = 2000
    data = {
        'valeur_vehicule_neuf': np.random.randint(80000, 700000, size=size),
        'age_vehicule_ans': np.random.randint(0, 15, size=size),
        'puissance_fiscale': np.random.choice([6, 7, 8, 9, 10, 11, 12], size=size, p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05]),
        'type_conducteur': np.random.choice(['Principal', 'Occasionnel'], size=size, p=[0.8, 0.2]),
        'zone_circulation': np.random.choice(['Urbaine', 'Rurale'], size=size, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    
    # Logique pour calculer la "charge du sinistre" (coût du sinistre)
    base_cost = df['valeur_vehicule_neuf'] * np.random.uniform(0.1, 0.3)
    age_factor = 1 - (df['age_vehicule_ans'] * 0.04)
    power_factor = 1 + (df['puissance_fiscale'] - 6) * 0.05
    zone_factor = df['zone_circulation'].map({'Urbaine': 1.1, 'Rurale': 0.9})
    
    df['charge_sinistre'] = base_cost * age_factor * power_factor * zone_factor
    df['charge_sinistre'] = df['charge_sinistre'].astype(int).clip(lower=2000)
    
    # Conversion des variables textuelles en nombres (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, columns=['type_conducteur', 'zone_circulation'], drop_first=True)
    
    # Définition des features (X) et de la cible (y)
    X = df_encoded.drop('charge_sinistre', axis=1)
    y = df_encoded['charge_sinistre']
    
    # Entraînement du modèle RandomForest
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Sauvegarde du modèle entraîné
    joblib.dump(model, MODEL_FILE)
    
    return model, list(X.columns)

# --- Chargement du modèle (ou entraînement si non existant) ---
# Note: Sur Streamlit Cloud, le modèle sera entraîné la première fois puis chargé depuis le fichier .pkl
if not os.path.exists(MODEL_FILE):
    st.info("Modèle non trouvé. Lancement de l'entraînement initial... (cela peut prendre un moment)")
    model, training_cols = train_model()
    st.success(f"Entraînement terminé ! Modèle sauvegardé sous `{MODEL_FILE}`.")
else:
    model = joblib.load(MODEL_FILE)
    # Définir les colonnes manuellement pour s'assurer de l'ordre correct
    training_cols = ['valeur_vehicule_neuf', 'age_vehicule_ans', 'puissance_fiscale', 'type_conducteur_Principal', 'zone_circulation_Urbaine']


# ==============================================================================
# PARTIE 2 : L'INTERFACE UTILISATEUR (STREAMLIT)
# ==============================================================================

st.set_page_config(page_title="Modélisation Charge Sinistre", layout="wide")

st.title("Modélisation de la Charge des Sinistres - Garantie Dommage")
st.markdown("Cette application est un **outil d'aide à la décision** pour les experts en assurance.")
st.markdown("---")

# --- Panneau latéral pour les entrées utilisateur ---
st.sidebar.header("Paramètres du Sinistre")

valeur_neuf = st.sidebar.slider("Valeur à neuf du véhicule (DH)", 80000, 1000000, 250000, 10000)
age_vehicule = st.sidebar.slider("Âge du véhicule (années)", 0, 20, 5)
puissance = st.sidebar.select_slider("Puissance fiscale (CV)", options=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
type_cond_text = st.sidebar.radio("Type de conducteur", ["Principal", "Occasionnel"])
zone_circ_text = st.sidebar.radio("Zone de circulation principale", ["Urbaine", "Rurale"])

if st.sidebar.button("Prédire la Charge du Sinistre", type="primary"):
    
    # Préparation des données d'entrée pour la prédiction
    input_data = {
        'valeur_vehicule_neuf': [valeur_neuf],
        'age_vehicule_ans': [age_vehicule],
        'puissance_fiscale': [puissance],
        'type_conducteur_Principal': [1 if type_cond_text == 'Principal' else 0],
        'zone_circulation_Urbaine': [1 if zone_circ_text == 'Urbaine' else 0]
    }
    input_df = pd.DataFrame(input_data)
    
    # S'assurer que les colonnes sont dans le même ordre que pour l'entraînement
    input_df = input_df.reindex(columns=training_cols, fill_value=0)

    # Prédiction avec le modèle chargé
    prediction = model.predict(input_df)
    charge_finale = int(prediction[0])
    
    # Affichage du résultat
    st.header("Résultat de la Modélisation")
    st.metric(label="Charge de Sinistre Prédite", value=f"{charge_finale:,} DH".replace(',', ' '))

    # Ajout de commentaires basés sur le résultat
    if charge_finale > 200000:
        st.error("Risque de charge élevée. Une expertise approfondie est recommandée.")
    elif charge_finale > 80000:
        st.warning("Charge moyenne à surveiller.")
    else:
        st.success("Charge standard estimée.")


else:
    st.info("Veuillez saisir les paramètres dans le panneau de gauche et cliquer sur 'Prédire'.")

st.markdown("---")
st.caption("Projet réalisé par HAFID HCHCHOUM - MATU ASSURANCE - 2025")
