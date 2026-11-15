# Documentation du simulateur CDFR 2027

Cette documentation s'adresse aux développeurs et aux assistants LLM qui doivent lire, modifier ou faire évoluer le projet. Elle décrit le code réel du dépôt, pas une cible théorique.

## Ordre de lecture

| Ordre | Document | À lire pour |
| --- | --- | --- |
| 1 | [`contexte-llm.md`](contexte-llm.md) | Comprendre le projet, les règles de travail et les invariants à ne pas casser. |
| 2 | [`architecture.md`](architecture.md) | Comprendre les flux d'exécution et les dépendances entre modules. |
| 3 | [`structure.md`](structure.md) | Localiser un symbole, un fichier ou une responsabilité. |
| 4 | [`domaine-et-regles.md`](domaine-et-regles.md) | Comprendre les unités, la géométrie, les actions et le score. |
| 5 | [`plan-modularisation.md`](plan-modularisation.md) | Préparer le découpage en modules interchangeables (proposition, pas encore implémenté). |

Si vous ne lisez qu'un seul fichier avant de modifier du code, lisez [`contexte-llm.md`](contexte-llm.md) puis la section concernée de [`structure.md`](structure.md).

## Hiérarchie des sources de vérité

En cas de contradiction, appliquer cet ordre :

1. Le code exécutable et les tests de `test/`.
2. `RobotRobotronik.toml` pour la géométrie du robot.
3. `AGENTS.md` pour les règles de travail du dépôt.
4. Cette documentation.
5. `README.md`, qui est la documentation utilisateur et peut résumer sans exhaustivité.

Les documents de `docs/` doivent être mis à jour dès qu'une responsabilité, un flux ou un contrat change.

## Conventions du dépôt

- Les commentaires, docstrings et messages d'erreur sont en français.
- Les identifiants Python sont en anglais, sauf les noms de modules historiques (`simulateur`, `reseau`, `match`, `interface`).
- Les coordonnées sont en millimètres, les durées en secondes et les angles du moteur en radians.
- Les degrés n'apparaissent que dans `RobotRobotronik.toml` (`angle_deg`) et dans l'affichage.
- L'axe X va vers la droite, l'axe Y vers le bas, l'origine est le coin supérieur gauche du tapis.
- Le paquet `src` n'a pas de `__init__.py` : c'est un paquet-espace de noms. Les modules s'exécutent avec `python -m src.<module>` depuis la racine.

## Commandes utiles

```bash
# Tests complets
python -B -m unittest discover -s test -v

# Match de référence, puis replay
python -B -m src.match
python -B -m src.match replay

# Interface manuelle
python -B -m src.interface

# Entraînement et comparaison
python -B -m src.reseau train --resume --matches 100
python -B -m src.reseau compare
```

Avec l'environnement fourni dans ce dépôt, remplacer `python` par `.venv/bin/python` ou activer `.venv`.

## État de cette documentation

Les documents `contexte-llm.md`, `architecture.md`, `structure.md` et `domaine-et-regles.md` décrivent l'état courant du code. Le document `plan-modularisation.md` est une proposition : il ne décrit pas du code déjà présent et ne doit pas être pris pour une API existante.

Pour vérifier la version décrite :

```bash
git log --oneline -1
git status --short
```
