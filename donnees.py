import pyreadstat
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def charger_donnees(chemin: str) -> pd.DataFrame:
    """
    Charge, nettoie et recode les données EDS Cameroun 2018.
    Le décorateur @st.cache_data évite de relire le fichier
    à chaque interaction utilisateur.
    """
    COLS = ["v005","v013","v025","v106","v130","v190","v501",
            "v157","v158","v159","v220","v313"]

    df, _ = pyreadstat.read_dta(chemin, usecols=COLS)

    # Nettoyage
    df = df[df["v313"].notna()]
    df = df[df["v013"] != 8]
    df = df[df["v130"] != 96]

    # Variable dépendante
    df["type_methode"] = pd.Categorical(
        df["v313"].map({0:"Aucune",2:"Traditionnelle",3:"Moderne"}),
        categories=["Aucune","Traditionnelle","Moderne"]
    )
    df = df[df["type_methode"].notna()]

    # Variables explicatives
    df["age_grp"] = pd.Categorical(
        df["v013"].map({1:"15-19",2:"20-24",3:"25-29",4:"30-34",
                        5:"35-39",6:"40-44",7:"45-49"}),
        categories=["15-19","20-24","25-29","30-34",
                    "35-39","40-44","45-49"], ordered=True
    )
    df["instruction"] = pd.Categorical(
        df["v106"].map({0:"Aucun",1:"Primaire",
                        2:"Secondaire",3:"Superieur"}),
        categories=["Aucun","Primaire","Secondaire","Superieur"],
        ordered=True
    )
    df["milieu"] = pd.Categorical(
        df["v025"].map({1:"Urbain",2:"Rural"}),
        categories=["Rural","Urbain"]
    )
    df["richesse"] = pd.Categorical(
        df["v190"].map({1:"Tres pauvre",2:"Pauvre",3:"Moyen",
                        4:"Riche",5:"Tres riche"}),
        categories=["Tres pauvre","Pauvre","Moyen","Riche","Tres riche"],
        ordered=True
    )
    df["religion"] = pd.Categorical(
        df["v130"].map({1:"Catholique",2:"Protestante",
                        3:"Autres chretiens",4:"Musulmane",
                        5:"Animiste",7:"Sans religion"}),
        categories=["Catholique","Protestante","Autres chretiens",
                    "Musulmane","Animiste","Sans religion"]
    )
    df["statut_mat"] = pd.Categorical(
        df["v501"].apply(lambda v:
            "En union" if v in [1,2] else
            "Jamais en union" if v == 0 else
            "Anciennement en union"
        ),
        categories=["En union","Jamais en union","Anciennement en union"]
    )
    df["parite"] = pd.Categorical(
        df["v220"].apply(lambda v:
            "Nullipare" if v == 0 else "Non nullipare"
        ),
        categories=["Nullipare","Non nullipare"]
    )
    df["expo_media"] = pd.Categorical(
        ((df["v157"]>=2)|(df["v158"]>=2)|(df["v159"]>=2))
        .map({True:"Exposee",False:"Non exposee"}),
        categories=["Non exposee","Exposee"]
    )
    df["poids"]   = df["v005"] / 1_000_000
    df["utilise"] = (df["type_methode"] != "Aucune").astype(int)

    VARS = ["type_methode","age_grp","statut_mat","parite",
            "instruction","richesse","milieu","religion",
            "expo_media","poids","utilise"]

    return df[VARS].dropna().copy()