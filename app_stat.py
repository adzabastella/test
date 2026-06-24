import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

import shap

# ── Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Contraception au Cameroun — EDS 2018",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COUL  = {"Aucune":"#E74C3C","Traditionnelle":"#F39C12","Moderne":"#27AE60"}
EMOJI = {"Aucune":"⛔","Traditionnelle":"🌿","Moderne":"💊"}
MSG   = {
    "Aucune":
        "D'après ce profil, la femme est **peu susceptible** "
        "d'utiliser une méthode contraceptive selon les données EDS 2018.",
    "Traditionnelle":
        "D'après ce profil, la femme est susceptible d'utiliser une "
        "**méthode traditionnelle** — abstinence périodique, retrait…",
    "Moderne":
        "D'après ce profil, la femme est susceptible d'utiliser une "
        "**méthode moderne** — pilule, DIU, injectable, implant…",
}
LABELS = {
    "age_grp_15-19":"Âge 15-19","age_grp_20-24":"Âge 20-24",
    "age_grp_25-29":"Âge 25-29","age_grp_30-34":"Âge 30-34",
    "age_grp_35-39":"Âge 35-39","age_grp_40-44":"Âge 40-44",
    "age_grp_45-49":"Âge 45-49",
    "statut_mat_En union":"En union",
    "statut_mat_Jamais en union":"Jamais en union",
    "statut_mat_Anciennement en union":"Anciennement en union",
    "parite_Nullipare":"Nullipare","parite_Non nullipare":"Non nullipare",
    "instruction_Aucun":"Sans instruction",
    "instruction_Primaire":"Instruction primaire",
    "instruction_Secondaire":"Instruction secondaire",
    "instruction_Superieur":"Instruction supérieure",
    "richesse_Tres pauvre":"Très pauvre","richesse_Pauvre":"Pauvre",
    "richesse_Moyen":"Richesse moyenne","richesse_Riche":"Riche",
    "richesse_Tres riche":"Très riche",
    "milieu_Rural":"Milieu rural","milieu_Urbain":"Milieu urbain",
    "religion_Catholique":"Catholique","religion_Protestante":"Protestante",
    "religion_Autres chretiens":"Autres chrétiens",
    "religion_Musulmane":"Musulmane","religion_Animiste":"Animiste",
    "religion_Sans religion":"Sans religion",
    "expo_media_Non exposee":"Non exposée","expo_media_Exposee":"Exposée",
}
FEATURES = ["age_grp","statut_mat","parite","instruction",
            "richesse","milieu","religion","expo_media"]
MODELES_ARBRES = ["Random Forest", "Gradient Boosting", "Extra Trees",
                   "Decision Tree", "AdaBoost"]

# ════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ════════════════════════════════════════════════════════════════
@st.cache_data
def charger_donnees():
    base = pd.read_csv(os.path.join(BASE_DIR, "base_eds2018.csv"))
    cats = {
        "type_methode": (["Aucune","Traditionnelle","Moderne"], False),
        "age_grp":      (["15-19","20-24","25-29","30-34",
                          "35-39","40-44","45-49"], True),
        "instruction":  (["Aucun","Primaire","Secondaire","Superieur"], True),
        "milieu":       (["Rural","Urbain"], False),
        "richesse":     (["Tres pauvre","Pauvre","Moyen",
                          "Riche","Tres riche"], True),
        "religion":     (["Catholique","Protestante","Autres chretiens",
                          "Musulmane","Animiste","Sans religion"], False),
        "statut_mat":   (["En union","Jamais en union",
                          "Anciennement en union"], False),
        "parite":       (["Nullipare","Non nullipare"], False),
        "expo_media":   (["Non exposee","Exposee"], False),
    }
    for col, (lst, ordered) in cats.items():
        if col in base.columns:
            base[col] = pd.Categorical(
                base[col], categories=lst, ordered=ordered)
    base["utilise"] = (base["type_methode"] != "Aucune").astype(int)
    return base

# ════════════════════════════════════════════════════════════════
# CALCUL DES ODDS RATIOS
# ════════════════════════════════════════════════════════════════
@st.cache_data
def calculer_or(_base):
    y = _base["type_methode"].cat.codes.values
    X = pd.get_dummies(_base[FEATURES], drop_first=True).astype(float)
    X = sm.add_constant(X, has_constant="add")

    m4 = MNLogit(y, X).fit(disp=False, method="bfgs", maxiter=1000)

    def extraire_or(idx):
        coef  = m4.params.iloc[:, idx]
        se    = m4.bse.iloc[:,    idx]
        p     = m4.pvalues.iloc[:, idx]
        OR    = np.exp(coef)
        ic_lo = np.exp(coef - 1.96*se)
        ic_hi = np.exp(coef + 1.96*se)
        sig   = np.where(p<0.001,"***",
                np.where(p<0.01,"**",
                np.where(p<0.05,"*","ns")))
        return pd.DataFrame({
            "OR": OR.round(3), "IC 2.5%": ic_lo.round(3),
            "IC 97.5%": ic_hi.round(3), "p-valeur": p.round(4), "Sig.": sig
        }, index=m4.model.exog_names)

    or_trad    = extraire_or(0)
    or_moderne = extraire_or(1)
    aic = round(m4.aic, 1)
    m_nul = MNLogit(y, np.ones((len(y),1))).fit(disp=False)
    mcf = round(1 - m4.llf / m_nul.llf, 4)
    return or_moderne, or_trad, aic, mcf

# ════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT ET COMPARAISON DE 7 MODÈLES ML
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def entrainer_et_comparer(_base):
    X = pd.get_dummies(_base[FEATURES], drop_first=False).astype(float)
    y = _base["type_methode"].cat.codes.values
    classes      = list(_base["type_methode"].cat.categories)
    feature_cols = X.columns.tolist()

    cv5   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    poids = compute_sample_weight("balanced", y)

    modeles = {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", class_weight="balanced",
            max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=30,
            class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=30,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=30,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.8, random_state=42),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=150, learning_rate=0.5, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=25),
    }

    resultats = []
    modeles_entraines = {}

    for nom, modele in modeles.items():
        if nom in ["Gradient Boosting", "AdaBoost"]:
            ba_list = []
            for tr, te in cv5.split(X.values, y):
                modele.fit(X.values[tr], y[tr], sample_weight=poids[tr])
                ba_list.append(balanced_accuracy_score(
                    y[te], modele.predict(X.values[te])))
            ba = np.mean(ba_list)
            modele.fit(X.values, y, sample_weight=poids)
        else:
            scores = cross_validate(modele, X.values, y, cv=cv5,
                                    scoring="balanced_accuracy", n_jobs=-1)
            ba = scores["test_score"].mean()
            modele.fit(X.values, y)

        resultats.append({"Modèle": nom, "Balanced Accuracy": round(ba, 4)})
        modeles_entraines[nom] = modele

    df_res = pd.DataFrame(resultats).sort_values(
        "Balanced Accuracy", ascending=False).reset_index(drop=True)
    meilleur_nom    = df_res.iloc[0]["Modèle"]
    meilleur_modele = modeles_entraines[meilleur_nom]

    return meilleur_modele, meilleur_nom, feature_cols, classes, df_res

# ════════════════════════════════════════════════════════════════
# CALCUL SHAP
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def calculer_shap(_model, _nom_modele, _X_sample_tuple, _feature_cols):
    X_sample = np.array(_X_sample_tuple)
    if _nom_modele in MODELES_ARBRES:
        explainer = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(X_sample)
    else:
        bg = shap.sample(X_sample, min(50, len(X_sample)))
        explainer = shap.KernelExplainer(_model.predict_proba, bg)
        sample_eval = shap.sample(X_sample, min(150, len(X_sample)))
        shap_values = explainer.shap_values(sample_eval)
        X_sample = sample_eval
    return shap_values, explainer, X_sample

# ── Prédiction ────────────────────────────────────────────────────
def predire(params, model, feature_cols, classes):
    df      = pd.DataFrame([params])
    dummies = pd.get_dummies(df, drop_first=False).astype(float)
    X       = dummies.reindex(columns=feature_cols, fill_value=0.0)
    probas  = model.predict_proba(X)[0]
    pred    = classes[int(np.argmax(probas))]
    return pred, probas, X

# ── Forest plot ───────────────────────────────────────────────────
def forest_plot(or_tab, titre):
    or_sig = or_tab[or_tab["Sig."] != "ns"].copy()
    or_sig = or_sig.sort_values("OR", ascending=True)
    or_sig.index = [LABELS.get(i, i) for i in or_sig.index]
    if or_sig.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3, len(or_sig)*0.42+1.5)))
    y_pos  = range(len(or_sig))
    colors = ["#E74C3C" if v<1 else "#27AE60" for v in or_sig["OR"]]
    ax.hlines(list(y_pos), or_sig["IC 2.5%"], or_sig["IC 97.5%"],
              color=colors, linewidth=2, alpha=0.7)
    ax.scatter(or_sig["OR"], list(y_pos), color=colors, s=55, zorder=5)
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    for i in range(len(or_sig)):
        ax.text(or_sig["IC 97.5%"].iloc[i]+0.05, i,
                f"{or_sig['OR'].iloc[i]:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(or_sig.index, fontsize=9)
    ax.set_xlabel("Odds Ratio (IC 95%)")
    ax.set_title(titre, fontweight="bold", fontsize=10)
    p1 = mpatches.Patch(color="#27AE60", label="OR > 1 (favorisant)")
    p2 = mpatches.Patch(color="#E74C3C", label="OR < 1 (frein)")
    ax.legend(handles=[p1, p2], fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_facecolor("#F8F9FA"); fig.patch.set_facecolor("#F8F9FA")
    plt.tight_layout()
    return fig

# ── Chargement / Entraînement ─────────────────────────────────────
base = charger_donnees()

with st.spinner("⚙️ Entraînement et comparaison de 7 modèles… (~20 secondes au premier lancement)"):
    model, nom_modele, feature_cols, classes, df_comparaison = \
        entrainer_et_comparer(base)
    or_mod, or_trad, aic_m4, mcf_m4 = calculer_or(base)

# Échantillon pour SHAP (limite le temps de calcul)
X_full = pd.get_dummies(base[FEATURES], drop_first=False).astype(float)
X_full = X_full.reindex(columns=feature_cols, fill_value=0.0)
X_sample_df = X_full.sample(min(800, len(X_full)), random_state=42)
X_sample_tuple = tuple(map(tuple, X_sample_df.values))

with st.spinner("Calcul des valeurs SHAP…"):
    shap_values, explainer, X_shap_eval = calculer_shap(
        model, nom_modele, X_sample_tuple, feature_cols)

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/4f/Flag_of_Cameroon.svg",
    width=70
)
st.sidebar.title("EDS Cameroun 2018")
st.sidebar.caption("Analyse de la contraception")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊  Statistiques & Résultats", "🎯  Prédiction"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Base :** {len(base):,} femmes  \n"
    f"**Modèle retenu :** {nom_modele}  \n"
    f"**Bal. Accuracy :** {df_comparaison.iloc[0]['Balanced Accuracy']:.4f}  \n"
    f"**Modèles comparés :** 7"
)

# ════════════════════════════════════════════════════════════════
# PAGE 1 — STATISTIQUES
# ════════════════════════════════════════════════════════════════
if page == "📊  Statistiques & Résultats":

    st.title("📊 Statistiques & Résultats")
    st.markdown("Analyse multivariée — EDS Cameroun 2018.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Effectif", f"{len(base):,}")
    c2.metric("Aucune méthode",
              f"{(base['type_methode']=='Aucune').sum():,}",
              f"{(base['type_methode']=='Aucune').mean()*100:.1f}%")
    c3.metric("Méthode moderne",
              f"{(base['type_methode']=='Moderne').sum():,}",
              f"{(base['type_methode']=='Moderne').mean()*100:.1f}%")
    c4.metric("Méthode traditionnelle",
              f"{(base['type_methode']=='Traditionnelle').sum():,}",
              f"{(base['type_methode']=='Traditionnelle').mean()*100:.1f}%")

    st.markdown("---")

    t1,t2,t3,t4 = st.tabs([
        "🔢 Analyse descriptive",
        "🔗 Analyse bivariée",
        "📈 Modélisation statistique",
        "🤖 Machine Learning & SHAP",
    ])

    # ── Onglet 1 ─────────────────────────────────────────────
    with t1:
        st.subheader("Distribution des variables")
        col1,col2 = st.columns([1,1])
        with col1:
            vc = base["type_methode"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5,5))
            ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                   colors=["#E74C3C","#F39C12","#27AE60"],
                   startangle=90,
                   wedgeprops={"edgecolor":"white","linewidth":2})
            ax.set_title("Type de méthode contraceptive (V313)",
                         fontweight="bold", pad=12)
            ax.set_facecolor("#F8F9FA"); fig.patch.set_facecolor("#F8F9FA")
            st.pyplot(fig); plt.close()
        with col2:
            pct = (vc/len(base)*100).round(1)
            st.markdown("**Effectifs par modalité**")
            st.dataframe(pd.DataFrame({"Effectif":vc,"%":pct}),
                         use_container_width=True)
            st.markdown(" ")
            st.markdown("**Par milieu de résidence (%)**")
            ct_m = (pd.crosstab(base["milieu"],base["type_methode"],
                                normalize="index")*100).round(1)
            st.dataframe(ct_m, use_container_width=True)

        st.markdown("---")
        var_map = {
            "Niveau d'instruction":"instruction","Groupe d'âge":"age_grp",
            "Indice de richesse":"richesse","Religion":"religion",
            "Statut matrimonial":"statut_mat",
            "Exposition aux médias":"expo_media",
            "Milieu de résidence":"milieu","Parité":"parite",
        }
        var_sel = st.selectbox("Explorer par variable :", list(var_map.keys()))
        col_sel = var_map[var_sel]
        ct = (pd.crosstab(base[col_sel],base["type_methode"],
                          normalize="index")*100)
        fig,ax = plt.subplots(figsize=(9,4))
        ct.plot(kind="bar", ax=ax,
                color=["#E74C3C","#F39C12","#27AE60"],
                edgecolor="black", width=0.7)
        ax.set_title(f"Utilisation contraceptive par {var_sel}",
                     fontweight="bold")
        ax.set_ylabel("Pourcentage (%)")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.legend(title="Méthode", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#F8F9FA"); fig.patch.set_facecolor("#F8F9FA")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Onglet 2 ─────────────────────────────────────────────
    with t2:
        st.subheader("Tests du Chi²")
        explicatives = {
            "Niveau d'instruction":"instruction","Groupe d'âge":"age_grp",
            "Milieu de résidence":"milieu","Indice de richesse":"richesse",
            "Religion":"religion","Statut matrimonial":"statut_mat",
            "Parité":"parite","Exposition aux médias":"expo_media",
        }
        recap = []
        for nom_v, col in explicatives.items():
            tab = pd.crosstab(base[col], base["type_methode"])
            chi2, p, ddl, _ = stats.chi2_contingency(tab)
            sig = ("***" if p<0.001 else "**" if p<0.01
                   else "*" if p<0.05 else "ns")
            recap.append({"Variable":nom_v,"Chi²":round(chi2,2),
                          "Ddl":ddl,"p-valeur":f"{p:.2e}","Sig.":sig})
        df_chi2 = pd.DataFrame(recap).sort_values("Chi²",ascending=False)

        col1,col2 = st.columns([1,1])
        with col1:
            st.markdown("**Récapitulatif**")
            st.dataframe(df_chi2, use_container_width=True, hide_index=True)
            st.caption("*** p<0,001  ** p<0,01  * p<0,05")
        with col2:
            fig,ax = plt.subplots(figsize=(6,5))
            ax.barh(df_chi2["Variable"][::-1], df_chi2["Chi²"][::-1],
                    color="#1A3A5C", edgecolor="black", alpha=0.85)
            ax.set_xlabel("Valeur du Chi²")
            ax.set_title("Chi² par variable", fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            ax.set_facecolor("#F8F9FA"); fig.patch.set_facecolor("#F8F9FA")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        var_biv = st.selectbox("Tableau croisé :", list(explicatives.keys()))
        col_biv = explicatives[var_biv]
        tab_pct = (pd.crosstab(base[col_biv], base["type_methode"],
                               normalize="index")*100).round(1)
        tab_eff = pd.crosstab(base[col_biv], base["type_methode"])
        chi2_v, p_v, _, _ = stats.chi2_contingency(tab_eff)
        sig_v = ("***" if p_v<0.001 else "**" if p_v<0.01
                 else "*" if p_v<0.05 else "ns")
        st.dataframe(tab_pct, use_container_width=True)
        st.markdown(
            f"**Chi²** = {chi2_v:.2f}  |  "
            f"**p** = {p_v:.2e}  |  {sig_v}")

    # ── Onglet 3 ─────────────────────────────────────────────
    with t3:
        st.subheader("Modèle logistique multinomial M4")
        col1,col2,col3 = st.columns(3)
        col1.metric("AIC", f"{aic_m4:,}")
        col2.metric("McFadden R²", f"{mcf_m4:.4f}")
        col3.metric("Observations", f"{len(base):,}")
        st.markdown("---")
        eq = st.radio("Équation :",
                      ["Moderne vs Aucune","Traditionnelle vs Aucune"],
                      horizontal=True)
        or_tab = (or_mod if eq=="Moderne vs Aucune" else or_trad)
        or_tab = or_tab.drop("const", errors="ignore")
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown(f"**Odds Ratios — {eq}**")
            st.dataframe(or_tab, use_container_width=True)
            st.caption("OR > 1 = favorisant  ·  OR < 1 = frein  "
                       "·  *** p<0,001  ** p<0,01  * p<0,05")
        with col2:
            fig = forest_plot(or_tab, f"Forest plot — {eq}")
            if fig:
                st.pyplot(fig); plt.close()

    # ── Onglet 4 — ML + SHAP ──────────────────────────────────
    with t4:
        st.subheader(f"Machine Learning — Modèle retenu : {nom_modele}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Modèle retenu", nom_modele)
        col2.metric("Balanced Accuracy",
                   f"{df_comparaison.iloc[0]['Balanced Accuracy']:.4f}")
        col3.metric("Modèles comparés", "7")

        st.info(
            "Sept algorithmes ont été entraînés et comparés en validation "
            "croisée 5-fold : Régression Logistique, Decision Tree, "
            "Random Forest, Extra Trees, Gradient Boosting, AdaBoost et KNN. "
            "Le modèle avec la meilleure Balanced Accuracy est retenu "
            "automatiquement."
        )

        st.markdown("---")
        st.markdown("### Comparaison des 7 modèles testés")

        col1, col2 = st.columns([1, 1.3])
        with col1:
            st.dataframe(df_comparaison, use_container_width=True,
                        hide_index=True)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors_bar = ["#27AE60" if i == 0 else "#1A3A5C"
                         for i in range(len(df_comparaison))]
            ax.bar(df_comparaison["Modèle"],
                  df_comparaison["Balanced Accuracy"],
                  color=colors_bar, edgecolor="black", alpha=0.85)
            ax.axhline(1/3, color="red", linestyle="--",
                      linewidth=1, label="Seuil aléatoire")
            ax.set_xticklabels(df_comparaison["Modèle"],
                              rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Balanced Accuracy")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.set_facecolor("#F8F9FA"); fig.patch.set_facecolor("#F8F9FA")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("### Importance des variables — SHAP")
        st.caption(
            "SHAP quantifie la contribution exacte de chaque variable à "
            "chaque prédiction, avec un signe — plus précis et plus "
            "explicable que l'importance Gini classique."
        )

        classe_shap = st.selectbox("Classe à expliquer :", classes,
                                    key="shap_classe_select")
        idx_classe = classes.index(classe_shap)

        try:
            sv = (shap_values[idx_classe] if isinstance(shap_values, list)
                  else shap_values[:, :, idx_classe])
            fig_shap, ax_shap = plt.subplots(figsize=(8, 6))
            plt.sca(ax_shap)
            shap.summary_plot(
                sv, X_shap_eval, feature_names=feature_cols,
                show=False, max_display=12
            )
            plt.title(f"SHAP — Variables influençant '{classe_shap}'",
                     fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()
        except Exception as e:
            st.warning(f"Affichage SHAP indisponible pour ce modèle : {e}")

        st.markdown("---")
        st.markdown("### Justification des choix méthodologiques")
        st.markdown("""
        | Choix | Justification |
        |-------|--------------|
        | **7 modèles comparés** | Sélection objective du meilleur, pas un choix a priori |
        | **Pas de fichier pkl** | Modèle entraîné dans l'app — aucun souci de version |
        | **Pas de standardisation** | Features = dummies 0/1 · arbres invariants à l'échelle |
        | **Pas de SMOTE** | 1 333 obs. Trad. suffisant · objectif inférentiel |
        | **sample_weight / class_weight** | Corrige le déséquilibre sans modifier les données |
        | **SHAP** | Explicabilité fine, par variable et par prédiction individuelle |
        """)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — PRÉDICTION
# ════════════════════════════════════════════════════════════════
elif page == "🎯  Prédiction":

    st.title("🎯 Prédiction du type de méthode contraceptive")
    st.markdown(
        f"Renseignez les caractéristiques d'une femme. Le modèle "
        f"**{nom_modele}** (meilleur parmi 7 testés) prédit : "
        "⛔ Aucune  ·  🌿 Traditionnelle  ·  💊 Moderne"
    )
    st.markdown("---")

    col_form, col_res = st.columns([1,1], gap="large")

    with col_form:
        st.subheader("Profil de la femme")
        age         = st.selectbox("Groupe d'âge",
                        ["15-19","20-24","25-29","30-34",
                         "35-39","40-44","45-49"])
        instruction = st.selectbox("Niveau d'instruction",
                        ["Aucun","Primaire","Secondaire","Superieur"])
        richesse    = st.selectbox("Indice de richesse",
                        ["Tres pauvre","Pauvre","Moyen","Riche","Tres riche"])
        milieu      = st.selectbox("Milieu de résidence",["Rural","Urbain"])
        religion    = st.selectbox("Religion",
                        ["Catholique","Protestante","Autres chretiens",
                         "Musulmane","Animiste","Sans religion"])
        statut      = st.selectbox("Statut matrimonial",
                        ["En union","Jamais en union",
                         "Anciennement en union"])
        parite      = st.selectbox("Parité",
                        ["Nullipare","Non nullipare"],
                        help="Nullipare = aucun enfant vivant")
        media       = st.selectbox("Exposition aux médias",
                        ["Non exposee","Exposee"],
                        help="Exposée = radio/TV/journal ≥ 1×/semaine")
        btn = st.button("🔍 Prédire", type="primary",
                        use_container_width=True)

    with col_res:
        st.subheader("Résultat de la prédiction")

        if btn:
            params = {
                "age_grp":age,"statut_mat":statut,"parite":parite,
                "instruction":instruction,"richesse":richesse,
                "milieu":milieu,"religion":religion,"expo_media":media,
            }
            prediction, probas, X_input = predire(
                params, model, feature_cols, classes)
            color     = COUL[prediction]
            confiance = max(probas) * 100

            st.markdown(
                f"""
                <div style="background:{color}18;border:2px solid {color};
                            border-radius:12px;padding:24px;
                            text-align:center;margin-bottom:18px;">
                  <div style="font-size:52px;">{EMOJI[prediction]}</div>
                  <div style="font-size:22px;font-weight:700;
                              color:{color};margin-top:10px;">
                    {prediction}</div>
                  <div style="font-size:30px;font-weight:800;
                              color:{color};margin-top:8px;">
                    {confiance:.1f}%</div>
                  <div style="font-size:12px;color:#666;margin-top:4px;">
                    probabilité estimée par {nom_modele}</div>
                </div>
                """, unsafe_allow_html=True
            )
            st.markdown(MSG[prediction])
            st.markdown("---")

            st.markdown("**Probabilités par classe :**")
            fig, ax = plt.subplots(figsize=(6,2.8))
            bars = ax.barh(classes, probas,
                           color=[COUL[c] for c in classes],
                           edgecolor="white", height=0.5)
            for bar, prob, cl in zip(bars, probas, classes):
                ax.text(min(prob+0.015,0.88),
                        bar.get_y()+bar.get_height()/2,
                        f"{prob*100:.1f}%", va="center",
                        fontsize=11, fontweight="bold", color=COUL[cl])
            ax.set_xlim(0,1)
            ax.axvline(1/3, color="gray", linestyle="--",
                       linewidth=0.9, alpha=0.5,
                       label="Seuil aléatoire (33%)")
            ax.set_facecolor("#F8F9FA"); fig.patch.set_facecolor("#F8F9FA")
            ax.spines[["top","right"]].set_visible(False)
            ax.set_xlabel("Probabilité"); ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # ── Explication SHAP individuelle ─────────────────
            st.markdown("---")
            st.markdown("**Explication SHAP de cette prédiction :**")
            st.caption(
                "Contribution de chaque facteur du profil à la "
                "prédiction — vert pousse vers la classe prédite, "
                "rouge s'y oppose."
            )

            try:
                idx_pred = classes.index(prediction)
                if nom_modele in MODELES_ARBRES:
                    shap_profil = explainer.shap_values(X_input.values)
                    if isinstance(shap_profil, list):
                        sv_profil = shap_profil[idx_pred][0]
                        base_val  = explainer.expected_value[idx_pred]
                    else:
                        sv_profil = shap_profil[0, :, idx_pred]
                        base_val  = explainer.expected_value[idx_pred]

                    fig_w, ax_w = plt.subplots(figsize=(8, 3.2))
                    plt.sca(ax_w)
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=sv_profil,
                            base_values=base_val,
                            data=X_input.values[0],
                            feature_names=[LABELS.get(f, f)
                                          for f in feature_cols]
                        ),
                        max_display=8, show=False
                    )
                    plt.tight_layout()
                    st.pyplot(fig_w)
                    plt.close()
                else:
                    st.caption(
                        f"Explication waterfall non disponible pour "
                        f"{nom_modele} (modèle non basé sur des arbres)."
                    )
            except Exception as e:
                st.caption(f"Explication SHAP indisponible pour ce profil.")

            # ── Facteurs actifs (importance globale) ──────────
            st.markdown("---")
            st.markdown("**Poids des facteurs du profil (importance globale) :**")
            if hasattr(model, "feature_importances_"):
                fi_all   = pd.Series(model.feature_importances_,
                                     index=feature_cols)
                actifs   = X_input.columns[
                    X_input.iloc[0]==1].tolist()
                fi_actif = fi_all[actifs].sort_values(
                    ascending=False).head(6)
                max_imp  = fi_actif.max() if not fi_actif.empty else 1

                for feat, imp in fi_actif.items():
                    label   = LABELS.get(feat, feat)
                    largeur = int(imp / max_imp * 100)
                    st.markdown(
                        f"""
                        <div style="display:flex;align-items:center;
                                    margin-bottom:9px;gap:12px;">
                          <span style="min-width:210px;font-size:13px;
                                       color:#333;">{label}</span>
                          <div style="flex:1;background:#E2E8F0;
                                      border-radius:6px;height:14px;">
                            <div style="width:{largeur}%;background:{color};
                                        height:14px;border-radius:6px;"></div>
                          </div>
                          <span style="font-size:12px;color:#888;
                                       min-width:48px;text-align:right;">
                            {imp*100:.2f}%</span>
                        </div>
                        """, unsafe_allow_html=True
                    )
            else:
                st.caption(
                    f"{nom_modele} n'a pas d'importance native — "
                    "voir l'explication SHAP ci-dessus."
                )

        else:
            st.info("👈 Renseignez le profil à gauche puis cliquez sur **Prédire**.")
            st.markdown(f"""
            **Les 3 classes prédites :**

            | Classe | Description |
            |--------|-------------|
            | ⛔ **Aucune** | Aucune méthode contraceptive |
            | 🌿 **Traditionnelle** | Abstinence périodique, retrait… |
            | 💊 **Moderne** | Pilule, DIU, injectable, implant… |

            **Modèle retenu :** {nom_modele} (meilleur parmi 7 testés)  
            **Entraîné sur :** 33 862 femmes (EDS 2018)  
            **Explicabilité :** SHAP (Shapley values)
            """)
