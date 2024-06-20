# SAE - S5 & S6
Groupe : BASQUIN Nicolas, BOLMONT Hugo, DAL GOBBO Théo

## Description du projet
Développement d'une interface de visualisation de prédiction de données sur 3 sujets:
<ul>
<li>éruptions solaires</li>
<li>météo</li>
<li>énergie<ul><li>production d'énergie</li><li>éolienne (détection d'anomalies)</li></ul>
</ul>
HTML/CSS/Bootstrap/Flask/JavaScript/Python/joblib/scikit-learn/Pandas/NumPy/Matplotlib
<br> modèles utlisés : LightGBM, CatBoost, XGBoost, RandomForest, ExtraTrees, DecisionTree
<br> tuning : Bayesian
<br> techniques d'ensemble utilisées : Voting, Stacking, Bagging

## Suivi du projet
    sklearn + node.js OU conversion scikit en tensorflow = pénible
    => utilisation de Flask

### 7/01/2024
    Nicolas : Initialisation du projet flask (création d'une première route et d'un premier template),
    Utilisation de joblib pour charger un modèle et l'utiliser afin de faire une prédiction via des caractéristiques données en paramètre depuis une page html

### 9-10/01/2024
    Théo : Création des routes / templates / de la structure du projet pour les 3 sujets,
    Réalisation du front avec Bootstrap,
    Bouton permettant de lancer un script qui enregistre les modèles en joblib
    Création d'une requête qui envoie en paramètre la source, le modèle ainsi qu'un nombre de période choisi, exécute un script python qui charge le modèle choisi, fait des prédictions et créer un graphique. Ce graphique est ensuite affiché sur la page html

### 11-12/01/2024
    Nicolas : page Solar Flares => affichage des informations concernant les données utilisées, ajout d'une sélection de modèle pour réaliser une prédiction en fonction des caractéristiques données par l'utilisateur, règlage des problèmes de compatibilité entre les modèles et Flask
    Hugo : page Weather => affichage des informations concernant les données utilisées, . . .
    Théo : page Energy Production => affichage des informations concernant les données utilisées, description du dataset, tableau des résultats des modèles RMSE / R-squared pour les deux sources "solar" et "wind", ajout d'un graphique présentant les données, rédaction des documents demandés dans le README.md (Suivi du projet / Architecture du projet / Indications générales de mise en œuvre), règlage des problèmes de compatibilité entre les modèles et Flask

## Architecture du projet
### Français
#### FRONTEND
    Le dossier templates contient les fichiers HTML qui sont utilisés pour afficher les pages web.
    CSS et Bootstrap sont utilisés pour le style des pages web.
    JQuery est également utilisé pour des requêtes AJAX.
    Ces pages web sont affichées par le serveur Flask et sont accessibles via les routes définies dans app.py.
    Des formulaires sont utilisés pour envoyer des requêtes POST au serveur Flask et des requêtes AJAX sont utilisées pour envoyer des requêtes GET au serveur Flask.
    Les pages web récupèrent les données / graphiques du serveur Flask et les affichent.
#### BACKEND
    Nous avons un serveur Flask app.py qui créer des routes pour des requêtes POST et GET / afficher des templates.
    Des requêtes exécutent des scripts Python qui chargent les données et enregistrent les modèles avec joblib dans les dossiers joblib/... .
    Les modèles sont ensuite utilisés pour prédire les données / créer des graphiques dans les dossiers static/images/... en fonction des caractéristiques données par l'utilisateur sur les pages web.
    Les données / graphiques sont ensuite renvoyés au serveur Flask et affichés sur les pages web.

### English
#### FRONTEND
    The "templates" folder contains HTML files used to display web pages.
    CSS and Bootstrap are employed for styling the web pages.
    JQuery is also utilized for AJAX requests.
    These web pages are rendered by the Flask server and can be accessed through the routes defined in app.py.
    Forms are used to send POST requests to the Flask server, and AJAX requests are used to send GET requests to the Flask server.
    The web pages retrieve data/graphs from the Flask server and display them.
#### BACKEND
    We have a Flask server in app.py that creates routes for handling POST and GET requests and rendering templates.
    Requests trigger Python scripts that load data and save models using joblib in the joblib/... directories.
    The models are then utilized to predict data/create graphs in the static/images/... directories based on user-provided features on the web pages.
    The data/graphs are then sent back to the Flask server and displayed on the web pages.

## Indications générales de mise en œuvre
### Français
#### Intallation des bonnes versions NumPy / SckiKit-Learn
    pip install --upgrade numpy==1.23.1
    pip install --upgrade scikit-learn==1.3.2
    pip install --upgrade joblib==1.3.2

#### Démarrer l'interface
    python3 app.py
#### Avant l'utilisation des formulaires de requêtes pour chaque sujet
    Exécuter les scripts Python qui enregistrent les modèles en joblib si un bouton y est dédié
    /!\ Ces actions peuvent prendre plusieurs minutes /!\
    Si aucun bouton n'est dédié à l'enregistrement des modèles, les modèles (peu lourds en espace de stockage) sont déjà enregistrés dans la structure du projet
#### Sélectionner / Définir les caractéristiques pour la requête choisie avant de l'exécuter

### English
#### Installing the correct versions of NumPy / SckiKit-Learn
    pip install --upgrade numpy==1.23.1
    pip install --upgrade scikit-learn==1.3.2
    pip install --upgrade joblib==1.3.2
#### Starting the interface
    python3 app.py
#### Before using the request forms for each subject
    Run the Python scripts that save the models using joblib if a button is dedicated to it
    /!\ These actions can take several minutes /!\
    If no button is dedicated to saving the models, the models (not very heavy in storage space) are already saved in the project structure
#### Select / Define the features for the chosen query before executing it