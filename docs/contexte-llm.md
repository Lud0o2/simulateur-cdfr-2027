# Contexte pour un assistant LLM

## But du projet

`simulateur-cdfr-2027` simule la table officielle Eurobot 2027 (Coupe de France de Robotique) pour :

- piloter un robot manuellement avec Tkinter ;
- jouer et rejouer des matchs automatiques ;
- entraîner un réseau de neurones qui choisit les prises, les dépôts et le retour à la salle du trône.

Le projet ne simule pas encore d'adversaire. Un seul robot évolue sur la table.

## Carte rapide du dépôt

| Chemin | Rôle |
| --- | --- |
| `src/app.py` | CLI unifiée : `match`, `replay`, `ui`, `train`, `compare`. |
| `src/composition/` | Profil, registre et composition par défaut. |
| `src/domaine/` | Modèle pur, géométrie, robot, scénario, score. |
| `src/terrains/` | Terrain officiel 2027. |
| `src/robots/` | Chargement TOML et `RobotGeometry`. |
| `src/regles/` | Barème, score 2027 et phases du match. |
| `src/physique/` | Moteur Pymunk, piles, freinage, préhenseur. |
| `src/planification/` | A*, planificateur direct, coût, rotation. |
| `src/actions/` | Prise, dépose, approche, retour, capacités. |
| `src/strategies/` | Stratégie de référence, lots, poses. |
| `src/deroulement/` | Déroulé du match. |
| `src/replay/` | Format, lecture, sauvegarde, lecteur. |
| `src/infra/` | Ressources, horloge, aléa, reporting, notification. |
| `src/reseaux/` | Caractéristiques, MLP, mutation, stockage. |
| `src/selection/` | Glouton, aléatoire, epsilon, politique réseau. |
| `src/recompenses/` | Variantes de récompense. |
| `src/evaluation/` | Scénarios, métriques, benchmark. |
| `src/ui/` | Thème, assets, rendu, clavier, fenêtre Tkinter, lecture des replays et options du tableau de bord. |
| `src/neat.py` | Neuroévolution NEAT, génomes, fitness et champions. |
| `src/interface_neat.py` | Tableau de bord NEAT et replay du champion. |
| `src/simulateur.py`, `src/match.py`, `src/reseau.py`, `src/interface.py` | Façades de compatibilité. |
| `test/test_simulateur.py` | Tests historiques. |
| `test/test_modularisation.py` | Tests des capacités, contrats et isolations. |
| `test/test_neat.py` | Tests de la neuroévolution et de `Playback`. |
| `RobotRobotronik.toml` | Contour, ventouses, capacités, mouvement et planificateur. |
| `assets/tapis_2027.png` | Image du tapis pour l'interface et les replays. |
| `AGENTS.md` | Règles de travail du dépôt. |
| `README.md` | Documentation utilisateur et feuille de route. |
| `docs/` | Cette documentation, destinée aux LLM et aux développeurs. |

## Invariants à respecter

1. **Le moteur reste utilisable sans fenêtre.** `src/physique/moteur.py` ne doit jamais importer `tkinter`, `PIL`, `src.ui` ni charger l'image du tapis.
2. **Le moteur n'attend pas le temps réel.** `Simulation.advance(seconds)` avance la physique à pas fixe (`PHYSICS_STEP = 1/480`). Un match de 100 s simulées doit pouvoir se jouer en quelques secondes.
3. **`reset()` rétablit l'état initial complet** : temps, robot, cap, ventouses, piles, mouvement, pivot, accumulateur de pas.
4. **Unités** : millimètres, secondes, radians dans le moteur. Les degrés restent dans le TOML et l'affichage.
5. **Repère** : origine au coin supérieur gauche, X vers la droite, Y vers le bas. À 0 rad, le robot regarde vers +X.
6. **La géométrie du robot vient du TOML**, jamais de constantes cachées. Les ventouses sont numérotées dans l'ordre des blocs `[[ventouses]]`.
7. **Les règles et collisions appartiennent au moteur**, pas à l'interface. L'interface ne fait que convertir des pixels en millimètres et appeler le moteur.
8. **Les paramètres physiques provisoires doivent rester signalés.** Masse du robot, inertie de pivot, frottements et accélérations n'ont pas encore été mesurés sur le robot réel.
9. **Toute modification du terrain ou des règles doit citer la section du règlement Eurobot utilisée.** Les références actuelles sont `D.2`, `D.3`, `D.4`, `E.3` et le plan `G.1`.
10. **Les tests existants doivent rester verts.** Commande de référence :

```bash
python -B -m unittest discover -s test -v
```

## Ce que fait le moteur, en une phrase par étape

- `Simulation.reset()` construit un espace Pymunk : bordures de table, obstacles fixes, robot cinématique, piles dynamiques.
- `move_to(x, y)` déplace le robot en ligne droite vers une destination, en s'arrêtant avant un obstacle fixe ou une bordure.
- `turn_by(angle)` tourne le robot ; si une pile libre gêne, le robot passe en dynamique et la pousse avec un pivot.
- `advance(seconds)` fait tourner la physique, capture les piles sous les ventouses armées et met à jour le temps.
- `score` calcule les points de construction des piles libres ; `return_score` calcule le bonus de salle du trône.

## Ce que fait le match automatique

- `play_match()` joue jusqu'à 100 s simulées.
- La construction s'arrête à `BUILD_DEADLINE = 90 s`.
- La stratégie de référence (`choose_plan`) choisit une pile, une ventouse et un dépôt par gain de score par seconde.
- Le mode multi-piles (`_play_batches`, utilisé dès qu'une politique est fournie) charge plusieurs ventouses avant de déposer.
- Le retour à la salle du trône vise 10 points (châssis entièrement dedans), puis 5 points (entrée partielle), avant 99,95 s.
- Le résultat est un dictionnaire de replay sérialisable en JSON.

## Ce que fait l'entraînement

- `reseau.features()` encode l'état du match et chaque candidat en 97 nombres.
- `Network` est un perceptron multicouche `97 → 16 → 8 → 1`, entraîné par Adam.
- `MatchPolicy.order()` classe les candidats avec le réseau ; il explore aussi aléatoirement pendant l'entraînement.
- `train_matches()` simule des matchs en parallèle, entraîne des copies du réseau, mute les poses de dépôt et garde le meilleur réseau.
- Le modèle est sauvegardé dans `reseau.npz`, les replays dans `replays/`.

## Pièges fréquents

- **Ne pas confondre `score` et `return_score`.** `score` ne compte que les piles libres dans les zones de construction ; `return_score` ne compte que la pose finale du châssis.
- **Ne pas compter une pile tenue.** `score_for_blocks` ignore les piles avec `grip is not None`.
- **Ne pas utiliser `pymunk` hors du moteur pour les règles.** Les helpers de collision sont dans `simulateur.py` et réutilisés par `match.py`.
- **Ne pas casser les caches.** `scoring_zones`, `_route`, `_visibility`, `_rotated_vertices` et d'autres fonctions sont mémoïsées ; leurs arguments doivent rester immuables et hachables.
- **Ne pas importer les symboles privés de `match.py` depuis un nouveau module sans raison.** `reseau.py` le fait déjà (`_action_state`, `_pickup_approach`, etc.), ce qui est un point de couplage à réduire.
- **Ne pas modifier `RobotRobotronik.toml` sans vérifier `test_robot_configuration`.** Le test par défaut vérifie un triangle de 380 mm et quatre ventouses.
- **Ne pas oublier l'accumulateur de pas.** `advance(0.01)` ne fait pas exactement 4 pas de 1/480 s à chaque appel ; le reste est conservé.
- **Ne pas supposer que le robot peut tourner sur place partout.** `can_turn_freely` et `turn_by` vérifient bordures, obstacles et piles libres.
- **Le robot ne peut construire que des murs aujourd'hui.** Les tours et les portes sont prévus mais le score et les actions correspondantes ne sont pas implémentés.

## Où modifier quoi

| Besoin | Fichier principal | Vérifications |
| --- | --- | --- |
| Changer la forme, les ventouses ou les capacités | `RobotRobotronik.toml` + `src/robots/loader.py` | `RobotConfigurationTest`, `TomlTest`, puis `python -B -m src.interface`. |
| Changer la physique ou les collisions | `src/physique/` | `SimulationTest`, puis match de référence. |
| Changer la planification | `src/planification/` | `AutomaticMatchTest`, `PlanningTest`. |
| Changer une action | `src/actions/` | `AutomaticMatchTest`. |
| Changer la stratégie ou les poses | `src/strategies/` | `AutomaticMatchTest`, `TrainingReplayTest`. |
| Changer la récompense ou la sélection | `src/recompenses/`, `src/selection/` | `RewardSelectionTest`, `TrainingReplayTest`. |
| Changer le réseau ou l'entraînement | `src/reseaux/`, `src/reseau.py` | `TrainingReplayTest`. |
| Changer NEAT ou son tableau de bord | `src/neat.py`, `src/interface_neat.py` | `test/test_neat.py`, puis `python -B -m src.neat verify`. |
| Changer l'affichage ou les commandes | `src/ui/` | Lancer l'interface, charger le tapis, tester les touches touchées. |
| Changer le terrain ou le score | `src/terrains/`, `src/regles/` | Citer le règlement, `SimulationTest` et `TomlTest`. |
| Changer le format de replay | `src/replay/` | Tests de replay et ouverture dans l'interface. |
| Changer la composition | `src/composition/` | `CompositionTest`. |

## Commandes de validation

```bash
# Toujours
python -B -m unittest discover -s test -v

# Match de référence sans fenêtre
python -B -m src.match

# Application unifiée
python -B -m src.app match
python -B -m src.app replay
python -B -m src.app ui

# Comparaison réseau / référence
python -B -m src.reseau compare

# NEAT : vérifier le dernier champion et ouvrir le tableau de bord
python -B -m src.neat verify
python -B -m src.neat replay

# Interface (vérification manuelle)
python -B -m src.interface
```

Le temps des 104 tests est d'environ 10 secondes sur la machine de référence. Une modification qui fait exploser ce temps doit être examinée.

## Glossaire

| Terme | Sens dans le code |
| --- | --- |
| Pile | Un corps rigide de trois pierres, rectangle de 320 × 110 mm vu du dessus. |
| Ventouse | Point de préhension du robot, défini dans le TOML par `position_mm` et `angle_deg`. |
| Prise | Saisie d'une pile par une ventouse, sur un grand côté. |
| Dépose | Relâchement d'une pile à une pose cible. |
| Rempart / zone de mur | Une des cinq zones de 400 × 200 mm où une pile rapporte un bonus de mur. |
| Tour | Cercle de rayon 100 mm autour d'un des quatre centres de tour. |
| Salle du trône | Rectangle `(0, 750, 500, 1250)` où le retour rapporte 5 ou 10 points. |
| Replay | JSON de frames d'affichage, avec obstacles, score, durée et métadonnées optionnelles. |
| Politique | Objet qui ordonne les candidats, généralement `MatchPolicy` autour d'un `Network`. |
| Candidat | Action envisageable : prise `(durée, pile, ventouse)` ou dépose `(gain, durée, ventouse, pose)`. |
