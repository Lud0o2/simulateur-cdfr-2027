# Règles du simulateur CDFR 2027

## Branches des étapes

- Avant de commencer une étape, créer `codex/etape-N` depuis `main`. Faire les modifications et les vérifications sur cette branche, puis avancer `main` après validation de l’utilisateur. Conserver la branche de chaque étape validée comme point de reprise.

## Architecture

- `src/physique/moteur.py` contient l’état du match et la physique Pymunk. `src/simulateur.py` est une façade de compatibilité. Le pilotage manuel, les tests et l’entraînement utilisent le même moteur, importable et avançable sans fenêtre, sans chargement du tapis et sans attente en temps réel.
- `src/ui/` gère le rendu Tkinter et les commandes ; `src/interface.py` est une façade. Les règles du jeu et les collisions appartiennent au moteur.
- `RobotRobotronik.toml` définit le contour, les ventouses, leur `construction` (`mur`, `tour`, `porte`), le `[mouvement]` et le `planificateur`. Utiliser cette géométrie pour les collisions, les prises et l’affichage, et déduire les commandes de la liste des ventouses. Une ventouse `tour` peut construire des murs ; une ventouse `porte` peut construire tours et murs.
- `src/domaine/`, `src/terrains/`, `src/robots/`, `src/regles/`, `src/planification/`, `src/actions/`, `src/strategies/`, `src/deroulement/`, `src/replay/`, `src/reseaux/`, `src/selection/`, `src/recompenses/` et `src/composition/` contiennent les modules interchangeables. Les anciens `simulateur.py`, `match.py`, `reseau.py` et `interface.py` restent des façades.
- `test/` contient les tests `unittest` et `assets/` les ressources d’affichage.
- `docs/` décrit l’architecture et la structure pour les LLM. Mettre ces documents à jour avec le code. Le plan de modularisation décrit l’état appliqué et les points restants.

## Simulation et vérification

- Garder les coordonnées en millimètres, les durées en secondes et les angles du moteur en radians. `advance(seconds)` avance la physique à pas fixe. `reset()` doit rétablir l’état initial, y compris pendant un déplacement ou une poussée.
- Pour une modification du terrain ou du jeu, citer la section du règlement Eurobot utilisée. Signaler les paramètres physiques provisoires en attendant leur mesure sur le robot réel.
- Depuis la racine, lancer `python -B -m unittest discover -s test -v`. Pour une modification de l’interface, vérifier aussi `python -B -m src.interface`, le chargement du tapis et les commandes touchées. Le README indique comment installer les dépendances et régénérer le tapis.
