# Fraude-Bancaire
 Détection de fraudes bancaires à l’aide du Machine Learning. Projet utilisant le classificateur SVM pour identifier les transactions suspectes à partir d’un dataset réel de cartes de crédit.
# Détection de Fraudes Bancaires avec SVM

Ce projet de Machine Learning a pour objectif de détecter les fraudes bancaires à partir du dataset public `creditcard.csv`. Il s'agit d'une implémentation simple mais efficace d'un pipeline de classification utilisant la méthode SVM (Support Vector Machine) avec noyau RBF.

## 📊 Dataset
Le dataset provient d'une base de données de transactions par carte de crédit effectuées en septembre 2013 par des titulaires de cartes européens. Il contient 284 807 transactions, dont 492 sont frauduleuses (0.17%).

## ⚙️ Fonctionnalités du script

- Chargement et exploration du dataset
- Visualisation des fraudes détectées
- Analyse de corrélations entre les variables
- Prétraitement et standardisation des données
- Entraînement d’un classificateur SVM
- Évaluation du modèle à l’aide de la courbe Précision-Rappel

## 📈 Résultat
Le modèle SVM atteint une **précision moyenne (Average Precision)** de **0.84**, ce qui montre une bonne performance dans la détection des fraudes.

## 🔧 Librairies utilisées

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn` (`model_selection`, `svm`, `metrics`, `preprocessing`)

## 🧠 Exécution
python :Fraude.py
/python Extension jupiter : Fraude.ipynb
