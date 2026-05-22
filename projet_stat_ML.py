import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import sys, os

sys.path.append(os.path.dirname(__file__))
from donnees import charger_donnees
from visualisations import (fig_distribution, fig_croise,
                                     fig_forest_plot)

#  Configuration de la page Streamlit 
st.set_page_config(
    page_title="Contraception au Cameroun — EDS 2018",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  Chargement des données et préparation 
#def get_data():
   # return charger_donnees("CMBR71FL.DTA")

#base = get_data()
base = charger_donnees()
#  Sidebar — navigation et filtres
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/4f/Flag_of_Cameroon.svg",
    width=80
)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choisir une section",
    ["🏠 Accueil",
     "📊 Analyse descriptive",
     "📈 Modélisation statistique",
     "🤖 Machine Learning",
     "📋 Rapport synthèse"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filtres")

milieu_filtre = st.sidebar.multiselect(
    "Milieu de résidence",
    options=["Rural", "Urbain"],
    default=["Rural", "Urbain"]
)
religion_filtre = st.sidebar.multiselect(
    "Religion",
    options=base["religion"].cat.categories.tolist(),
    default=base["religion"].cat.categories.tolist()
)

# Application des filtres
base_filtre = base[
    base["milieu"].isin(milieu_filtre) &
    base["religion"].isin(religion_filtre)
].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**{len(base_filtre):,}** femmes sélectionnées"
    f" sur {len(base):,}"
)

# PAGE 1 — ACCUEIL

if page == "🏠 Accueil":
    st.title("🇨🇲 Déterminants de la contraception au Cameroun")
    st.subheader("Enquête Démographique et de Santé — EDS 2018")

    st.markdown("""
    Cette application présente les résultats d'une analyse multivariée
    des déterminants de l'utilisation contraceptive au Cameroun,
    réalisée à partir des données de l'EDS 2018.

    **Question de recherche :**
    > *Quels facteurs sociodémographiques, économiques et culturels
    > sont associés au type de méthode contraceptive utilisée
    > par les femmes en âge de reproduction au Cameroun en 2018 ?*

    **Méthodes utilisées :**
    - Régression logistique multinomiale (M1 → M4)
    - Régression logistique binaire complémentaire
    - Random Forest et Gradient Boosting (Machine Learning)
    """)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Effectif analysé", f"{len(base):,}")
    col2.metric("Sans méthode",
                f"{(base['type_methode']=='Aucune').sum():,}",
                f"{(base['type_methode']=='Aucune').mean()*100:.1f}%")
    col3.metric("Méthode moderne",
                f"{(base['type_methode']=='Moderne').sum():,}",
                f"{(base['type_methode']=='Moderne').mean()*100:.1f}%")
    col4.metric("Méthode traditionnelle",
                f"{(base['type_methode']=='Traditionnelle').sum():,}",
                f"{(base['type_methode']=='Traditionnelle').mean()*100:.1f}%")


# PAGE 2 — ANALYSE DESCRIPTIVE

elif page == "📊 Analyse descriptive":
    st.title("📊 Analyse descriptive")

    st.info(f"Données filtrées : **{len(base_filtre):,}** femmes")

    tab1, tab2 = st.tabs(["Distribution", "Tableaux croisés"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(fig_distribution(base_filtre))
        with col2:
            vc  = base_filtre["type_methode"].value_counts().sort_index()
            pct = (vc / len(base_filtre) * 100).round(1)
            st.dataframe(
                pd.DataFrame({"Effectif":vc,"Pourcentage (%)":pct}),
                use_container_width=True
            )

    with tab2:
        variable = st.selectbox(
            "Choisir une variable explicative",
            {
                "Groupe d'âge":          "age_grp",
                "Niveau d'instruction":  "instruction",
                "Milieu de résidence":   "milieu",
                "Indice de richesse":    "richesse",
                "Religion":              "religion",
                "Statut matrimonial":    "statut_mat",
                "Exposition aux médias": "expo_media",
            }.keys()
        )
        var_map = {
            "Groupe d'âge":          "age_grp",
            "Niveau d'instruction":  "instruction",
            "Milieu de résidence":   "milieu",
            "Indice de richesse":    "richesse",
            "Religion":              "religion",
            "Statut matrimonial":    "statut_mat",
            "Exposition aux médias": "expo_media",
        }
        col_sel = var_map[variable]

        st.pyplot(fig_croise(base_filtre, col_sel,
                             f"Utilisation contraceptive par {variable}"))

        tab_eff = pd.crosstab(base_filtre[col_sel],
                              base_filtre["type_methode"])
        tab_pct = (tab_eff.div(tab_eff.sum(axis=1), axis=0)*100).round(1)
        chi2, p, _, _ = stats.chi2_contingency(tab_eff)
        sig = "***" if p<0.001 else ("**" if p<0.01 else
              ("*" if p<0.05 else "ns"))

        st.dataframe(tab_pct, use_container_width=True)
        st.markdown(
            f"**Test du Chi²** : χ² = {chi2:.2f} | "
            f"p = {p:.2e} | {sig}"
        )

# PAGE 3 — MODÉLISATION STATISTIQUE

elif page == "📈 Modélisation statistique":
    st.title("📈 Modélisation statistique — Logit multinomial M4")

    st.markdown("""
    Le modèle M4 est le modèle final retenu. Il inclut l'ensemble
    des variables explicatives sociodémographiques, socioéconomiques,
    culturelles et médiatiques.
    """)

    @st.cache_data
    def estimer_modele(_base):
        def encoder(data, covs):
            X = pd.get_dummies(data[covs], drop_first=True).astype(float)
            return sm.add_constant(X, has_constant="add")

        covs_M4 = ["age_grp","statut_mat","parite","instruction",
                   "richesse","milieu","religion","expo_media"]
        y    = _base["type_methode"].cat.codes.values
        X    = encoder(_base, covs_M4)
        M_nul = MNLogit(y, np.ones((len(y),1))).fit(
            disp=False, method="bfgs", maxiter=500)
        M4   = MNLogit(y, X).fit(
            disp=False, method="bfgs", maxiter=1000)
        return M4, M_nul, X

    with st.spinner("Estimation du modèle en cours..."):
        M4, M_nul, X_enc = estimer_modele(base)

    mcf = 1 - M4.llf / M_nul.llf
    col1, col2, col3 = st.columns(3)
    col1.metric("AIC", f"{M4.aic:.1f}")
    col2.metric("McFadden R²", f"{mcf:.4f}")
    col3.metric("Observations", f"{len(base):,}")

    st.markdown("---")

    categorie = st.radio(
        "Catégorie à afficher",
        ["Moderne vs Aucune", "Traditionnelle vs Aucune"],
        horizontal=True
    )
    idx = 1 if categorie == "Moderne vs Aucune" else 0

    coef  = M4.params.iloc[:, idx]
    se    = M4.bse.iloc[:,    idx]
    p     = M4.pvalues.iloc[:, idx]
    OR    = np.exp(coef)
    ic_lo = np.exp(coef - 1.96*se)
    ic_hi = np.exp(coef + 1.96*se)
    sig   = np.where(p<0.001,"***",
            np.where(p<0.01,"**",
            np.where(p<0.05,"*","ns")))

    or_tab = pd.DataFrame({
        "OR":OR.round(3),"IC 2.5%":ic_lo.round(3),
        "IC 97.5%":ic_hi.round(3),"p-valeur":p.round(4),"Sig.":sig
    }, index=M4.model.exog_names).drop("const", errors="ignore")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(or_tab, use_container_width=True)
    with col2:
        st.pyplot(fig_forest_plot(or_tab, f"Forest plot — {categorie}"))


# PAGE 4 — MACHINE LEARNING
elif page == "🤖 Machine Learning":
    st.title("🤖 Extension Machine Learning")

    FEATURES = ["age_grp","statut_mat","parite","instruction",
                "richesse","milieu","religion","expo_media"]

    @st.cache_data
    def entrainer_ml(_base):
        X = pd.get_dummies(
            _base[FEATURES], drop_first=True).astype(float)
        y_multi  = _base["type_methode"].cat.codes.values
        y_binary = _base["utilise"].values

        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=30,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        ba = cross_val_score(rf, X.values, y_multi,
                             cv=cv5, scoring="balanced_accuracy")
        auc = cross_val_score(rf, X.values, y_binary,
                              cv=cv5, scoring="roc_auc")
        rf.fit(X.values, y_multi)

        fi = pd.DataFrame({
            "Variable":  X.columns,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=False)

        return ba, auc, fi

    with st.spinner("Entraînement Random Forest (patience...)"):
        ba_cv, auc_cv, fi_df = entrainer_ml(base)

    col1, col2, col3 = st.columns(3)
    col1.metric("Balanced Accuracy CV-5",
                f"{ba_cv.mean():.4f}", f"±{ba_cv.std():.4f}")
    col2.metric("AUC-ROC binaire CV-5",
                f"{auc_cv.mean():.4f}", f"±{auc_cv.std():.4f}")
    col3.metric("Nombre de features", len(fi_df))

    st.markdown("---")
    st.subheader("Importance des variables (Random Forest)")

    n_vars = st.slider("Nombre de variables à afficher", 5, 20, 12)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, n_vars*0.4+1))
    fi_plot = fi_df.head(n_vars)
    ax.barh(fi_plot["Variable"][::-1],
            fi_plot["Importance"][::-1],
            color="steelblue", edgecolor="black")
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Importance des variables — Random Forest",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(fi_df.head(n_vars), use_container_width=True)

# PAGE 5 — RAPPORT SYNTHÈSE

elif page == "📋 Rapport synthèse":
    st.title("📋 Rapport de synthèse")

    st.markdown("""
    ### Principaux résultats

    **78,5 %** des femmes camerounaises en âge de reproduction
    n'utilisent aucune méthode contraceptive en 2018.

    #### Déterminants de l'utilisation des méthodes modernes
    """)

    resultats = pd.DataFrame({
        "Variable":    ["Instruction supérieure","Instruction secondaire",
                        "Instruction primaire","Richesse très riche",
                        "Religion animiste","Religion musulmane",
                        "Milieu urbain","Exposition aux médias"],
        "OR":          [5.62, 4.23, 3.71, 2.48,
                        0.39, 0.54, 1.24, 1.11],
        "Sig.":        ["***","***","***","***",
                        "***","***","***","**"],
        "Interprétation": [
            "5,6× plus de chances qu'une femme sans instruction",
            "4,2× plus de chances qu'une femme sans instruction",
            "3,7× plus de chances qu'une femme sans instruction",
            "2,5× plus de chances qu'une femme très pauvre",
            "61% moins de chances qu'une catholique",
            "46% moins de chances qu'une catholique",
            "24% plus de chances qu'en milieu rural",
            "11% plus de chances si exposée aux médias",
        ]
    })

    st.dataframe(resultats, use_container_width=True, hide_index=True)

    st.markdown("""
    #### Vérification des hypothèses

    | Hypothèse | Verdict |
    |-----------|---------|
    | H1 — Instruction → méthodes modernes | ✅ Confirmée *** |
    | H2 — Milieu urbain → accès élevé | ✅ Confirmée (médiation partielle) |
    | H3 — Richesse → utilisation accrue | ✅ Confirmée *** |
    | H4 — Médias → adoption moderne | ⚠️ Partiellement confirmée ** |
    | H5 — Religion → modération | ✅ Fortement confirmée *** |
    | H6 — Parité élevée → besoin fort | ✅ Confirmée (sens inverse) |

    #### Performance des modèles

    | Modèle | Métrique | Score |
    |--------|----------|-------|
    | Logit multinomial M4 | McFadden R² | 0.095 |
    | Random Forest multinomial | Balanced Accuracy CV-5 | 0.553 |
    | Random Forest binaire | AUC-ROC CV-5 | 0.738 |
    """)
