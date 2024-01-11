# Project Architecture

## Français
### FRONTEND
    Le dossier templates contient les fichiers HTML qui sont utilisés pour afficher les pages web.
    CSS et Bootstrap sont utilisés pour le style des pages web.
    JQuery est également utilisé pour des requêtes AJAX.
    Ces pages web sont affichées par le serveur Flask et sont accessibles via les routes définies dans app.py.
    Des formulaires sont utilisés pour envoyer des requêtes POST au serveur Flask et des requêtes AJAX sont utilisées pour envoyer des requêtes GET au serveur Flask.
    Les pages web récupèrent les données / graphiques du serveur Flask et les affichent.
### BACKEND
    Nous avons un serveur Flask app.py qui créer des routes pour des requêtes POST et GET / afficher des templates.
    Des requêtes exécutent des scripts Python qui chargent les données et enregistrent les modèles avec joblib dans les dossiers joblib/... .
    Les modèles sont ensuite utilisés pour prédire les données / créer des graphiques dans les dossiers static/images/... en fonction des caractéristiques données par l'utilisateur sur les pages web.
    Les données / graphiques sont ensuite renvoyés au serveur Flask et affichés sur les pages web.
<br>

## English
### FRONTEND
    The "templates" folder contains HTML files used to display web pages.
    CSS and Bootstrap are employed for styling the web pages.
    JQuery is also utilized for AJAX requests.
    These web pages are rendered by the Flask server and can be accessed through the routes defined in app.py.
    Forms are used to send POST requests to the Flask server, and AJAX requests are used to send GET requests to the Flask server.
    The web pages retrieve data/graphs from the Flask server and display them.
### BACKEND
    We have a Flask server in app.py that creates routes for handling POST and GET requests and rendering templates.
    Requests trigger Python scripts that load data and save models using joblib in the joblib/... directories.
    The models are then utilized to predict data/create graphs in the static/images/... directories based on user-provided features on the web pages.
    The data/graphs are then sent back to the Flask server and displayed on the web pages.