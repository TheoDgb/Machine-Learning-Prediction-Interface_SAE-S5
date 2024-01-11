# SAE-S5

### Démarrer l'interface
    python3 app.py

## Description du projet
Développement d'une interface de visualisation de prédiction de données sur 3 sujets:
<ul><li>éruptions solaires</li><li>météo</li><li>production d'énergie</li></ul>
html/css/bootstrap/flask/javascript/python/joblib/scikit-learn/pandas/numpy/matplotlib
<br> modèles utlisés : LightGBM, CatBoost, XGBoost, RandomForest, ExtraTrees, DecisionTree
<br> tuning : Bayesian
<br> techniques d'ensemble utilisées : Voting, Stacking, Bagging
<br><br> Basquin Nicolas, Bolmont Hugo, Dal Gobbo Théo

## Suivi du projet
    sklearn + node.js OU conversion scikit en tensorflow = pénible
    => utilisation de Flask

### 7/01/2024
    Nicolas : Initialisation du projet flask (création d'une première route et d'un premier template),
    Utilisation de joblib pour charger un modèle et l'utiliser afin de faire une prédiction via des caractéristiques données en paramètre depuis une page html

### 9-10-11/01/2024
    Théo : Création des routes / templates / de la structure du projet pour les 3 sujets,
    Réalisation du front avec Bootstrap,
    Script qui enregistre les modèles en joblib
    Création d'une requête qui envoie en paramètre la source, le modèle ainsi qu'un nombre de période choisi, exécute un script python qui charge le modèle choisi, fait des prédictions et créer un graphique. Ce graphique est ensuite affiché sur la page html 