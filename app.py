import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# PARTIE 1 : CHARGEMENT DU MODÈLE (SANS ENTRAÎNEMENT)
# ==============================================================================

# Le modèle est déjà entraîné et présent dans le repository GitHub.
# On a juste besoin de le charger.
try:
    model = joblib.load('charge_sinistre_model.pkl')
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop() # Arrête l'application si le modèle ne peut pas être chargé

# Définir les colonnes manuellement pour s'assurer de l'ordre correct
# C'est l'ordre exact que le modèle attend.
TRAINING_COLS = [
    'valeur_vehicule_neuf', 
    'age_vehicule_ans', 
    'puissance_fiscale', 
    'type_conducteur_Principal', 
    'zone_circulation_Urbaine'
]


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
    input_df = input_df.reindex(columns=TRAINING_COLS, fill_value=0)

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
# Remplace les informations ci-dessous par les tiennes
st.caption("Projet réalisé par Hafid Hchchoum - EST Agadir - 2025")
