import pyreadstat
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def charger_donnees():
    base = pd.read_csv("base_eds2018.csv")
    base["type_methode"] = pd.Categorical(
        base["type_methode"],
        categories=["Aucune","Traditionnelle","Moderne"])
    base["age_grp"] = pd.Categorical(
        base["age_grp"],
        categories=["15-19","20-24","25-29","30-34",
                    "35-39","40-44","45-49"], ordered=True)
    base["instruction"] = pd.Categorical(
        base["instruction"],
        categories=["Aucun","Primaire","Secondaire","Superieur"],
        ordered=True)
    base["milieu"] = pd.Categorical(
        base["milieu"], categories=["Rural","Urbain"])
    base["richesse"] = pd.Categorical(
        base["richesse"],
        categories=["Tres pauvre","Pauvre","Moyen",
                    "Riche","Tres riche"], ordered=True)
    base["religion"] = pd.Categorical(
        base["religion"],
        categories=["Catholique","Protestante","Autres chretiens",
                    "Musulmane","Animiste","Sans religion"])
    base["statut_mat"] = pd.Categorical(
        base["statut_mat"],
        categories=["En union","Jamais en union",
                    "Anciennement en union"])
    base["parite"] = pd.Categorical(
        base["parite"], categories=["Nullipare","Non nullipare"])
    base["expo_media"] = pd.Categorical(
        base["expo_media"], categories=["Non exposee","Exposee"])
    return base

@st.cache_data
def charger_resultats():
    or_mod  = pd.read_csv("or_moderne.csv",        index_col=0)
    or_trad = pd.read_csv("or_traditionnelle.csv", index_col=0)
    fi      = pd.read_csv("importance_rf.csv")
    return or_mod, or_trad, fi
