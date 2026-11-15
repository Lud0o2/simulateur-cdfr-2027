# Structure du code

## Arborescence

```text
simulateur-cdfr-2027/
├── AGENTS.md
├── README.md
├── RobotRobotronik.toml
├── requirements.txt
├── assets/
│   └── tapis_2027.png
├── docs/
│   ├── README.md
│   ├── contexte-llm.md
│   ├── architecture.md
│   ├── structure.md
│   ├── domaine-et-regles.md
│   └── plan-modularisation.md
├── replays/                  (ignoré par git)
├── reseau.npz                (ignoré par git)
├── src/
│   ├── simulateur.py
│   ├── match.py
│   ├── reseau.py
│   └── interface.py
└── test/
    └── test_simulateur.py
```

Il n'y a ni `pyproject.toml`, ni `setup.py`, ni `src/__init__.py`. `src` fonctionne comme un paquet-espace de noms, ce qui suffit pour `python -m src.match` et pour `import src.match` depuis la racine.

## `src/simulateur.py`

Responsabilité : état du match et physique Pymunk. C'est le seul module qui connaît Pymunk.

### Constantes

| Symbole | Valeur | Rôle |
| --- | --- | --- |
| `TABLE_WIDTH` | `3000` | Largeur de la table en mm. |
| `TABLE_HEIGHT` | `2000` | Hauteur de la table en mm. |
| `SPEED` | `1000` | Vitesse maximale de translation, mm/s. |
| `DRIVE_ACCELERATION` | `1000` | Accélération de translation, mm/s². |
| `TURN_STEP` | `radians(15)` | Pas de rotation manuelle. |
| `TURN_SPEED` | `radians(360)` | Vitesse de rotation, rad/s. |
| `TURN_ACCELERATION` | `radians(1200)` | Accélération de pivot, rad/s², provisoire. |
| `ROBOT_FILE` | `RobotRobotronik.toml` à la racine | Fichier de géométrie par défaut. |
| `START` | `(250.0, 1000.0)` | Position initiale du centre du robot. |
| `THRONE_ROOM` | `(0, 750, 500, 1250)` | Rectangle de la salle du trône. |
| `TABLE` | quatre coins | Bordure physique. |
| `START_BLOCKS` | 10 rectangles | Piles initiales, voir [`domaine-et-regles.md`](domaine-et-regles.md#terrain). |
| `CONTACT_GAP` | `1e-6` | Marge anti-chevauchement, mm. |
| `PHYSICS_STEP` | `1/480` | Pas physique fixe, s. |
| `ROBOT_MASS` | `5.0` | Masse effective provisoire, kg. |
| `PILE_MASS` | `0.09` | Masse d'une pile, kg. |
| `FLOOR_DECELERATION` | `4200` | Freinage de glissement des piles, mm/s². |
| `FLOOR_ANGULAR_DECELERATION` | `10` | Freinage de rotation des piles, rad/s². |
| `SUCTION_REACH` | `5` | Tolérance de contact de la ventouse, mm. |
| `SUCTION_ALIGNMENT` | `cos(radians(20))` | Alignement minimal des faces. |
| `WALL_ZONES` | 5 polygones | Remparts jaunes, chacun 400 × 200 mm. |
| `TOWER_CENTERS` | 4 points | Centres des tours. |
| `TOWER_RADIUS` | `100` | Rayon de tour, mm. |

### Chargement du robot

| Symbole | Rôle |
| --- | --- |
| `RobotGeometry` | Dataclass gelée : `contour` et `mounts`. `mounts` contient `(position_mm, vecteur_de_face)`. |
| `load_robot(path=ROBOT_FILE)` | Lit le TOML, valide le contour convexe et les ventouses, renvoie `RobotGeometry`. Mémoïsée par `functools.cache`. |
| `_finite_number` | Vérifie un nombre fini de type `int` ou `float`. |
| `_read_point` | Valide une paire `[x, y]`. |

Le chargeur refuse :

- un champ inconnu ;
- un `contour_mm` de moins de trois sommets ;
- un contour non convexe, avec sommets alignés ou dupliqués ;
- un contour qui dépasse de la table à la position de départ ;
- une ventouse sans `position_mm` ou `angle_deg`, ou avec un angle non fini.

### Géométrie et collisions

| Symbole | Rôle |
| --- | --- |
| `edges(points)` | Paires de sommets consécutifs, fermées. |
| `rectangle_corners(rectangle)` | Convertit `(left, top, right, bottom)` en quatre coins. |
| `travel_time(distance)` | Durée idéale d'une translation à l'arrêt vers l'arrêt, avec accélération trapézoïdale. |
| `_wall_zone(x, y, angle=0)` | Construit un rempart de 400 × 200 mm tourné. |
| `_world_corners(shape)` | Coins monde d'un `pymunk.Poly`. |
| `_polygon_bounds(polygon)` | Boîte englobante `(min_x, min_y, max_x, max_y)`, mémoïsée. |
| `polygon_contact(...)` | Fraction du trajet où deux polygones se chevauchent. |
| `contact_axes(corners, footprint)` | Axes SAT et bornes projetées. |
| `swept_contact(position, delta, axes, ...)` | Balayage SAT d'un polygone en translation. |
| `table_contact(position, delta, footprint)` | Fraction avant sortie de table. |
| `arc_hits_edge(point, angle, start, end)` | Détecte le balayage d'un sommet en rotation. |
| `throne_room_score(position, footprint)` | `10` si le châssis est entièrement dedans, `5` s'il chevauche, `0` sinon. |
| `_rotated_vertices(vertices, angle)` | Rotation mémoïsée. |
| `_footprint_radius(footprint)` | Rayon maximal, mémoïsé. |

### Modèle de données

| Symbole | Champs | Usage |
| --- | --- | --- |
| `Movement` | `end`, `blocked`, `turn`, `rotation_blocked` | Dernier ordre de déplacement et son état. |
| `Block` | `position`, `angle`, `corners` | Vue immuable d'une pile. |
| `SuctionCup` | `position`, `enabled` | Vue immuable d'une ventouse. |
| `_Grip` | `cup`, `shape`, `angle` | Pile tenue par une ventouse. |
| `_Pile` | `body`, `shape`, `floor`, `size`, `grip`, `_block` | État physique interne d'une pile. |
| `RobotGeometry` | `contour`, `mounts` | Géométrie chargée du TOML. |

`_Pile.free_block()` met en cache le `Block` tant que la pose ne change pas.

### Score

| Symbole | Rôle |
| --- | --- |
| `scoring_zones(block)` | Renvoie `(indices_de_remparts, scoring)`. `scoring` vaut `True` dans un rempart ou une tour. Mémoïsée. |
| `free_wall_slots(blocks)` | Nombre de remparts encore libres. |
| `construction_score(sites)` | `3` points par pile qui marque, `5` points par rempart distinct occupé. |

Le calcul des remparts utilise un appariement glouton avec réaffectation : deux piles qui se disputent la même zone ne comptent qu'un bonus de mur. Le `TODO` du code indique que les portes et les tours ne sont pas encore comptées finement.

### `Simulation`

| Membre | Rôle |
| --- | --- |
| `__init__(obstacles=(), *, blocks=START_BLOCKS, robot=None)` | Configure les obstacles fixes, les piles initiales et la géométrie. |
| `reset()` | Rétablit temps, mouvement, pivot, espace Pymunk, robot, ventouses et piles. |
| `heading` | Cap courant en radians. |
| `footprint` | Contour du robot tourné, relatif au centre. |
| `moving` | `True` si le corps du robot est dynamique. |
| `position` | Centre du robot. |
| `blocks` | Tuple de `Block` pour toutes les piles, tenues comprises. |
| `score` | Score de construction des piles libres. |
| `score_for_blocks(blocks)` | Score pour une liste de poses donnée. |
| `return_score` | Bonus de salle du trône, `0` si le robot bouge. |
| `held_pile(cup)` | Index de la pile tenue par une ventouse, ou `None`. |
| `can_turn_freely(angle)` | Vérifie bordures, obstacles et piles libres. |
| `rotation_clear(position, footprints, angle, polygons)` | Version de classe utilisée par la planification. |
| `suction_cups` | Positions monde et états des ventouses. |
| `toggle_suction(number)` | Coupe, relâche ou arme une ventouse, puis tente la capture. |
| `grip_pile(number, index)` | Saisie directe à l'arrêt, utilisée par les actions. |
| `footprints()` | Générateur : robot + piles tenues, en repère relatif au centre. |
| `move_to(x, y)` | Déplacement en ligne droite, avec pré-vérification statique. |
| `turn_by(angle)` | Rotation en roulant, pivot dynamique ou rotation immédiate. |
| `advance(seconds)` | Avance la physique à pas fixe et capture les piles armées. |

Méthodes internes importantes : `_capture_piles`, `_capture`, `_static_travel`, `_stop`, `_step`, `_stalled`, `_step_turn`, `_set_heading`, `_turn_crosses_border`, `_turn_hits`, `_block`.

## `src/match.py`

Responsabilité : planification de trajets, actions de prise/dépose, stratégies, replays et CLI de match.

### Constantes

| Symbole | Valeur | Rôle |
| --- | --- | --- |
| `MATCH_SECONDS` | `100` | Durée maximale simulée. |
| `BUILD_DEADLINE` | `90` | Fin des constructions. |
| `FRAME_STEP` | `0.05` | Pas d'enregistrement et de suivi. |
| `PICKUP_STANDOFF` | `25` | Distance d'alignement avant la prise, mm. |
| `PICKUP_STROKE` | `50` | Course maximale vers la pile, mm. |
| `PICKUP_GAP` | `2` | Jeu au contact, mm. |
| `DROP_RETREAT` | `50` | Recul après dépose, mm. |
| `ROUTE_MARGIN` | `5` | Marge de route autour des obstacles, mm. |
| `REPLAY_DIR` | `replays/` | Dossier des essais. |
| `DROP_SITES` | 45 | 5 remparts × grille 3 × 3. |
| `CENTERED_DROPS` | 5 | Centre et axe de chaque rempart. |
| `ALIGNED_ANGLES` | 5 zéros | Angles alignés par défaut. |
| `DROP_ANGLES` | 18 | De -90° à +80° par pas de 10°. |
| `ALTERNATE_DROPS` | 360 | 5 remparts × 4 décalages × 18 angles. |
| `WAYPOINTS` | 56 | Grille 8 × 7 de points de passage. |
| `WAYPOINT_TIMES` | table | Durées de translation entre points. |
| `OPEN_SPACE` | `(1450, 1000)` | Point central de dégagement. |

### Données de planification

| Symbole | Champs |
| --- | --- |
| `Approach` | `duration`, `before`, `turn`, `after`, `rolling`. |
| `Pickup` | `approach`, `end`. |
| `Plan` | `pile`, `cup`, `pickup`, `drop`, `retreat`. |

### Routeur

| Symbole | Rôle |
| --- | --- |
| `_clear` / `_clear_bounded` | Vérifie qu'un segment reste libre pour une ou plusieurs empreintes. |
| `_planned_axes` | Axes SAT avec marge. |
| `_visibility` | Fabrique une fonction de visibilité mémoïsée. |
| `_obstacle_waypoints` | Coins de contournement autour des obstacles. |
| `_corner_bypasses` | Points de passage entre deux coins. |
| `_route` | A* sur un graphe de visibilité, minimise le temps de translation. |
| `_turn_envelope` | Rectangle couvrant le balayage d'une rotation. |
| `_pivot_clear` | Vérifie un pivot sur place. |
| `_approach` | Choisit une rotation en roulant ou un pivot, puis un trajet. |

Le routeur est un A* avec points de passage. Il utilise `_segment_time` et `WAYPOINT_TIMES` pour estimer les coûts. Les commentaires `shortcut:` signalent des simplifications à étendre.

### Actions

| Symbole | Rôle |
| --- | --- |
| `_follow(simulation, approach, ...)` | Exécute un `Approach` pas à pas. |
| `_action_state(simulation)` | Position, empreintes et obstacles pour une action. |
| `take_pile(simulation, pile, cup, ...)` | Prise complète : approche, saisie, course. |
| `_take_pickup(...)` | Variante utilisant un `Pickup` déjà calculé. |
| `_stroke_end(...)` | Fin de course limitée par la table et les obstacles. |
| `_pickup_approach(...)` | Calcule les deux approches possibles d'une pile pour une ventouse. |
| `_wall_drop_zones(block)` | Vérifie qu'une pose de dépôt touche un rempart. |
| `_drop_retreat(...)` | Recul après dépose. |
| `_drop_target(...)` | Pose du robot pour amener la pile à la pose cible. |
| `_drop_approach(...)` | Trajet vers une dépose. |
| `place_pile(simulation, cup, x, y, angle, ...)` | Dépose, vérifie la pose, relâche et recule. |

`_pickup_approach` gère le décalage latéral quand deux ventouses de même axe pourraient se gêner. Il renvoie l'approche la plus rapide parmi les deux grands côtés.

### Stratégies

| Symbole | Rôle |
| --- | --- |
| `choose_plan(simulation, unavailable)` | Heuristique gloutonne gain / durée pour un dépôt à la fois. |
| `_play_reference(simulation, record)` | Joue la stratégie de référence. |
| `_play_batches(simulation, record, policy, drop_poses)` | Charge plusieurs ventouses, puis dépose ; accepte une politique. |
| `_frame(simulation, label)` | Construit une frame de replay. |
| `_block_shape`, `_shape_key`, `_drop_pose_table` | Cache les poses de dépôt compatibles avec une forme de pile. |
| `_return_home(simulation, record)` | Cherche une entrée à 10 ou 5 points avant le buzzer. |
| `play_match(simulation, *, batches=False, policy=None, record_frames=True)` | Orchestre le match et renvoie le replay. |

### Replay et CLI

| Symbole | Rôle |
| --- | --- |
| `save_replay(replay, directory=REPLAY_DIR, *, descriptive=False)` | Écrit un essai numéroté de façon atomique. |
| `_trials(directory)` | Indexe les fichiers `essai-*.json`. |
| `Replay` | Valide un replay et expose `frame` selon `time`. |
| `completion_beep()` | Bip de fin. |
| `main()` | `python -m src.match` ou `python -m src.match replay [numéro]`. |

## `src/reseau.py`

Responsabilité : caractéristiques, réseau MLP, politique et entraînement.

### Constantes

| Symbole | Valeur | Rôle |
| --- | --- | --- |
| `LAYERS` | `(97, 16, 8, 1)` | Architecture du MLP. |
| `TABLE_DIAGONAL` | `hypot(3000, 2000)` | Normalisation des distances. |
| `ACTION_DISTANCE` | `84` | Indice de la ligne de distance dans les entrées. |
| `MODEL` | `reseau.npz` | Modèle par défaut. |
| `PICKUP_LIMIT` | `8` | Nombre de prises classées par le réseau. |
| `EXPLORATION_INTERVAL` | `8` | Un match d'exploration tous les 8 matchs. |
| `PROGRESS_WINDOW` | `50` | Patience avant redémarrage en mode sans fin. |
| `DROP_OFFSET_LIMIT` | `200` | Décalage maximal d'une pose de dépôt, mm. |
| `BATCH_SIZE` | `64` | Taille de lot d'apprentissage. |
| `EPOCHS` | `8` | Budget d'époques par mise à jour. |
| `MAX_STEPS` | `64` | Plafond de pas d'apprentissage. |
| `LEARNING_RATE` | `0.002` | Pas initial. |
| `LEARNING_RATE_FLOOR` | `0.0002` | Pas minimal. |
| `LEARNING_RATE_HALVING` | `200` | Matchs entre deux divisions du pas. |
| `ADAM_BETAS` | `(0.9, 0.999)` | Paramètres Adam. |
| `ADAM_EPSILON` | `1e-8` | Stabilité Adam. |
| `MOMENT_NAMES` | quatre noms | Clés de sauvegarde des moments Adam. |

### Fonctions et classes

| Symbole | Rôle |
| --- | --- |
| `features(simulation, held, choices, *, pickup, pickup_plans=None)` | Encode l'état et les candidats en `(N, 97)`. |
| `_candidate_count(choices, pickup)` | Limite les prises à `PICKUP_LIMIT`. |
| `Network` | MLP NumPy, Adam, sauvegarde/chargement `.npz`, mémoire d'expérience. |
| `MatchPolicy` | Classe les candidats avec le réseau, gère l'exploration et `observe`. |
| `_replay_rank(replay)` | `(score, -durée)`. |
| `_validated_replay(network, rank, seed)` | Rejoue un match pour vérifier le rang. |
| `_simulate_match(job)` | Fonction de worker pour les processus. |
| `_mutate_drop_pose(network, rng)` | Mute un angle ou un décalage de dépôt. |
| `_training_workers(count)` | Nombre de processus. |
| `train_matches(...)` | Entraînement borné, validation, comparaison à l'incumbent. |
| `_train_endless(...)` | Entraînement par séries. |
| `_train_matches(...)` | Boucle générationnelle : simulation, mutation, apprentissage, sélection. |
| `main()` | CLI `train` et `compare`. |

### Format du modèle

Le fichier `.npz` contient :

- les poids et biais de chaque couche (`arr_0` à `arr_5`) ;
- les moments Adam (`adam_moment_w_N`, etc.) ;
- `drop_offsets` et `drop_angles` ;
- `experience` : lignes `(ouverture_1, ouverture_2, rang_score, rang_temps, entrées..., cible)`, vide si aucune ;
- `training_matches` et `adam_updates`.

`Network.load` accepte les anciens modèles à 96 entrées : il insère une ligne neutre à `ACTION_DISTANCE` et décale les indices. Il valide formes, types et valeurs finies.

## `src/interface.py`

Responsabilité : rendu Tkinter, commandes manuelles et lecture des replays.

| Symbole | Rôle |
| --- | --- |
| `create_window(simulation=None, *, replay=None, trial=None)` | Construit la fenêtre. Importe `tkinter` et `PIL` à l'intérieur. |
| `draw()` | Met à jour score, piles, robot, flèche avant, ventouses, cible et textes. |
| `fit_board(event)` | Calcule `scale` et `border`, redimensionne le tapis. |
| `click(event)` | Clic sur une ventouse ou destination. |
| `nudge(dx, dy)` | Déplacement clavier de 100 mm. |
| `press`, `release`, `release_all` | Gestion des répétitions clavier. |
| `reset_match()` | `simulation.reset()` puis redessine. |
| `tick()` | `simulation.advance(now - last_frame)` puis `draw()`. |
| Contrôles replay | Bouton pause/reprise/relance, curseur, glisser-déposer. |

Commandes manuelles :

- clic sur le tapis : destination ;
- flèches : ±100 mm ;
- `Q` / `D` : ±15° ;
- `1` à `9` et `0` : ventouses, plus les équivalents AZERTY et pavé numérique ;
- `Échap` : quitter le plein écran ; `F11` : basculer ;
- `Reset match` : réinitialiser.

L'interface ne contient aucune règle de jeu ni aucun calcul de collision.

## `test/test_simulateur.py`

Un seul fichier `unittest`, quatre classes, 78 tests.

| Classe | Nombre | Couverture |
| --- | --- | --- |
| `TrainingReplayTest` | 25 | Réseau, Adam, sauvegarde/chargement, politique, exploration, classement des replays, CLI. |
| `SimulationTest` | 32 | Translation, rotation, collisions, poussées, reset, ventouses, prises/déposes, score. |
| `AutomaticMatchTest` | 18 | Routeur, dépôts, stratégie multi-piles, replays, retour. |
| `RobotConfigurationTest` | 3 | TOML par défaut, rectangle personnalisé, contour concave refusé. |

Le fichier définit `reference_simulation` et `two_pile_simulation` avec une géométrie fixe de triangle de 380 mm, indépendante du TOML courant. Cela garde les scénarios reproductibles après édition du fichier robot.

## `RobotRobotronik.toml`

Schéma actuel :

```toml
# Contour convexe, sommets dans l'ordre du périmètre, en mm.
contour_mm = [
    [-219.39310229205776, 0],
    [109.69655114602888, -190],
    [109.69655114602888, 190],
]

# Une ventouse par bloc, numérotée dans l'ordre.
[[ventouses]]
position_mm = [109.69655114602888, -100]
angle_deg = 0
```

Contraintes :

- `contour_mm` : au moins trois sommets, convexes, sans alignement, dans la table à la position de départ.
- `ventouses` : liste optionnelle ; chaque bloc doit contenir exactement `position_mm` et `angle_deg`.
- L'angle est en degrés, 0° vers +X.
- Le nombre de ventouses pilote les touches et les boucles du match.

## Caches et performances

| Cache | Clé | Pourquoi |
| --- | --- | --- |
| `load_robot` | chemin | Évite de relire le TOML. |
| `_polygon_bounds` | polygone | Boîtes englobantes répétées. |
| `_rotated_vertices` | sommets, angle | Rotations répétées. |
| `_footprint_radius` | empreinte | Rayon répété. |
| `scoring_zones` | `Block` | Score et planification. |
| `_segment_time` | deux points | Coûts de route. |
| `_rotated_footprint` | empreinte, angle | Routeur. |
| `_planned_axes` | coins, empreinte, marge | SAT. |
| `_visibility` | empreintes, obstacles, marge | Fonction de visibilité. |
| `_obstacle_waypoints` | empreintes, obstacles | Points de contournement. |
| `_route` | départ, arrivée, empreintes, obstacles, limite | A*. |
| `_pivot_clear` | position, empreintes, angle, polygones | Pivots. |
| `_approach` | départ, arrivée, angle, empreintes, obstacles | Trajets. |

Les clés doivent rester immuables. Un objet Pymunk ou une liste modifiable dans une clé de cache provoquerait des erreurs ou des résultats périmés.

## Importations croisées à connaître

```python
# src/match.py
from .simulateur import (
    CONTACT_GAP, DRIVE_ACCELERATION, PHYSICS_STEP, START, TABLE_HEIGHT, TABLE_WIDTH,
    THRONE_ROOM, TOWER_CENTERS, TOWER_RADIUS, TURN_SPEED, WALL_ZONES, Block, Simulation,
    _finite_number, _polygon_bounds as _bounds, construction_score, contact_axes,
    polygon_contact, rectangle_corners, scoring_zones, swept_contact, table_contact,
    throne_room_score, travel_time,
)

# src/reseau.py
from .match import (
    ALIGNED_ANGLES, CENTERED_DROPS, DROP_ANGLES, MATCH_SECONDS, REPLAY_DIR, _action_state,
    _drop_target, _pickup_approach, _rotated_footprint, _visibility, _wall_drop_zones,
    completion_beep, play_match, save_replay,
)
from .simulateur import (
    Block, START_BLOCKS, TABLE_HEIGHT, TABLE_WIDTH, WALL_ZONES, Simulation,
    free_wall_slots, scoring_zones,
)
```

Toute réorganisation doit traiter ces imports : ils sont la principale raison pour laquelle les modules ne sont pas interchangeables aujourd'hui.

## TODO et raccourcis présents dans le code

- `construction_score` : compter les portes et les tours quand le moteur saura les représenter.
- `_approach` : un seul changement de cap par trajet, à étendre pour les passages plus étroits.
- `_route` : points de passage échantillonnés, à densifier pour les accès étroits.
- `_return_home` : cinq poses par cap, à élargir si les accès sont encombrés.
- `MatchPolicy.order` : le réseau classe les huit prises les plus rapides, à élargir si elles bloquent l'exploration.
- `README` : étape 7 (murs et portes) et étape 8 (duels) ne sont pas implémentées.
