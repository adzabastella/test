import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COULEURS3 = ["#E74C3C","#F39C12","#27AE60"]

def fig_distribution(base):
    fig, ax = plt.subplots(figsize=(7, 4))
    vc    = base["type_methode"].value_counts().sort_index()
    bars  = ax.bar(vc.index, vc.values,
                   color=COULEURS3, edgecolor="black", width=0.6)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x()+b.get_width()/2,
                b.get_height()+100,
                f"{v:,}\n({v/len(base)*100:.1f}%)",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Distribution de la méthode contraceptive",
                 fontweight="bold")
    ax.set_ylabel("Effectif")
    ax.set_ylim(0, vc.max()*1.25)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

def fig_croise(base, variable, titre):
    ct = (pd.crosstab(base[variable], base["type_methode"],
                      normalize="index") * 100)
    fig, ax = plt.subplots(figsize=(8, 4))
    ct.plot(kind="bar", ax=ax, color=COULEURS3,
            edgecolor="black", width=0.7)
    ax.set_title(titre, fontweight="bold")
    ax.set_ylabel("Pourcentage (%)")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="Méthode", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

def fig_forest_plot(or_table, titre):
    or_sig = or_table[or_table["Sig."] != "ns"].copy()
    or_sig = or_sig.sort_values("OR", ascending=True)

    noms = {
        "age_grp_20-24":"Âge 20-24","age_grp_25-29":"Âge 25-29",
        "age_grp_30-34":"Âge 30-34","age_grp_35-39":"Âge 35-39",
        "age_grp_40-44":"Âge 40-44","age_grp_45-49":"Âge 45-49",
        "statut_mat_Jamais en union":"Jamais en union",
        "statut_mat_Anciennement en union":"Anc. en union",
        "parite_Non nullipare":"Non nullipare",
        "instruction_Primaire":"Instr. Primaire",
        "instruction_Secondaire":"Instr. Secondaire",
        "instruction_Superieur":"Instr. Supérieur",
        "richesse_Pauvre":"Richesse Pauvre",
        "richesse_Moyen":"Richesse Moyen",
        "richesse_Riche":"Richesse Riche",
        "richesse_Tres riche":"Richesse Très riche",
        "milieu_Urbain":"Urbain",
        "religion_Protestante":"Protestante",
        "religion_Autres chretiens":"Autres chrét.",
        "religion_Musulmane":"Musulmane",
        "religion_Animiste":"Animiste",
        "religion_Sans religion":"Sans religion",
        "expo_media_Exposee":"Exposée médias",
    }
    or_sig.index = [noms.get(i,i) for i in or_sig.index]

    fig, ax = plt.subplots(figsize=(9, len(or_sig)*0.42+1.5))
    y_pos  = range(len(or_sig))
    colors = ["#E74C3C" if v<1 else "#27AE60" for v in or_sig["OR"]]

    ax.hlines(list(y_pos), or_sig["IC 2.5%"], or_sig["IC 97.5%"],
              color=colors, linewidth=2, alpha=0.7)
    ax.scatter(or_sig["OR"], list(y_pos),
               color=colors, s=55, zorder=5)
    ax.axvline(1, color="black", linestyle="--", linewidth=1)

    for i in range(len(or_sig)):
        ax.text(or_sig["IC 97.5%"].iloc[i]+0.05, i,
                f"{or_sig['OR'].iloc[i]:.2f}",
                va="center", fontsize=8)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(or_sig.index, fontsize=9)
    ax.set_xlabel("Odds Ratio (IC 95%)")
    ax.set_title(titre, fontweight="bold", fontsize=10)

    p1 = mpatches.Patch(color="#27AE60", label="OR > 1 (facteur favorisant)")
    p2 = mpatches.Patch(color="#E74C3C", label="OR < 1 (facteur défavorisant)")
    ax.legend(handles=[p1,p2], fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig