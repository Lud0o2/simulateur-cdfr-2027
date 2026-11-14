# Règles du simulateur CDFR 2027

## Architecture

- `src/simulateur.py` contient l’état du match et la physique Pymunk. Le pilotage manuel, les tests et le futur entraînement utilisent le même moteur. Le moteur doit pouvoir être importé et avancé sans fenêtre, sans chargement du tapis et sans attente en temps réel.
- `src/interface.py` gère le rendu Tkinter, les commandes manuelles et la conversion des pixels en millimètres. Les règles du jeu et les collisions appartiennent au moteur.
- `RobotRobotronik.toml` définit le contour et les ventouses du robot. Utiliser cette géométrie pour les collisions, les prises et l’affichage, et déduire les commandes de la liste des ventouses.
- `test/` contient les tests `unittest`, `assets/` les ressources d’affichage et `include/` est réservé aux éventuels en-têtes natifs.

## Simulation et vérification

- Garder les coordonnées en millimètres, les durées en secondes et les angles du moteur en radians. `advance(seconds)` avance la physique à pas fixe. `reset()` doit rétablir l’état initial, y compris pendant un déplacement ou une poussée.
- Pour une modification du terrain ou du jeu, citer la section du règlement Eurobot utilisée. Signaler les paramètres physiques provisoires en attendant leur mesure sur le robot réel.
- Depuis la racine, lancer `python -B -m unittest discover -s test -v`. Pour une modification de l’interface, vérifier aussi `python -B -m src.interface`, le chargement du tapis et les commandes touchées. Le README indique comment installer les dépendances et régénérer le tapis.
