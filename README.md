# Bank Marketing Prediction 🏦

Prédiction du succès d'une campagne marketing bancaire par classification binaire.

## Contexte

Ce projet s'appuie sur le [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
(45 211 clients, 16 variables). L'objectif est de prédire si un client va souscrire à un dépôt
à terme (`y = yes/no`) suite à un appel téléphonique.

**Type de problème** : Classification binaire déséquilibrée (~88% non / ~12% oui)

---

## Pipeline
```
Découverte → EDA univariée/bivariée/multivariée → Feature Engineering → Modélisation → Optimisation
```

### Étapes clés
- Identification et exclusion du **data leakage** (`duration`)
- Gestion du déséquilibre via `class_weight='balanced'` et **SMOTE**
- Encodage différencié : ordinal (education, month), binaire (default, housing), one-hot (job, poutcome...)
- Feature engineering : `contacted_before`, `balance_log`, `season`, `day_period`
- Comparaison de 4 modèles + optimisation par `RandomizedSearchCV`

---

## Résultats

| Modèle | ROC-AUC |
|---|---|
| **Gradient Boosting** ✅ | **0.803** |
| Random Forest | 0.790 |
| Decision Tree | 0.760 |
| Régression Logistique | 0.750 |

**Métrique principale** : ROC-AUC (contexte déséquilibré → l'accuracy est trompeuse)

---

## Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![pandas](https://img.shields.io/badge/pandas-2.x-blue)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-green)
```bash
pip install -r requirements.txt
```

---

## Structure
```
bank-marketing-prediction/
├── bank_marketing_prediction.ipynb   # Notebook principal
├── bank-full.csv                     # Dataset complet (45k lignes)
├── bank.csv                          # Dataset réduit (4.5k lignes)
├── requirements.txt
└── README.md
```

---

## Lancer le projet
```bash
git clone https://github.com/ton-username/bank-marketing-prediction
cd bank-marketing-prediction
pip install -r requirements.txt
jupyter notebook bank_marketing_prediction.ipynb
```

---

## Auteur

**Bahani** — [LinkedIn](https://linkedin.com/in/ton-profil) · [Portfolio](https://ton-portfolio.com)

*Projet académique — Epitech Lyon, MSc Data Science & Business Intelligence*
```

---

Et le `requirements.txt` minimal :
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
jupyter
