# Architecture

## Vue d'ensemble

Le code est désormais réparti en paquets spécialisés. Les anciens modules
`simulateur.py`, `match.py`, `reseau.py` et `interface.py` restent des façades
de compatibilité pour les tests et les commandes existantes.

```text
app.py / cli
  -> composition (config, registre, défaut)
       -> robots + terrains + regles
       -> physique
       -> planification
       -> actions
       -> strategies
       -> recompenses + selection + reseaux
       -> deroulement + replay
       -> ui
```

| Paquet | Rôle | Dépend de |
| --- | --- | --- |
| `domaine` | Modèle, géométrie pure, robot, scénario, score. | Bibliothèque standard. |
| `terrains` | Données du terrain officiel 2027. | `domaine`. |
| `robots` | Chargement TOML et vue `RobotGeometry`. | `domaine`, `terrains`, Pymunk. |
| `regles` | Barème, score 2027, phases du match. | `domaine`, `terrains`. |
| `physique` | Moteur Pymunk, piles, freinage, préhenseur, collisions. | `domaine`, `regles`, `robots`, `terrains`. |
| `planification` | A*, planificateur direct, coût, rotation. | `domaine`, `physique`, `terrains`. |
| `actions` | Prise, dépose, approche, retour, capacités. | `planification`, `domaine`. |
| `strategies` | Référence, lots, catalogue de poses. | `actions`, `planification`. |
| `deroulement` | Déroulé du match et enregistrement. | `strategies`, `actions`, `replay`. |
| `replay` | Format, lecture, sauvegarde, lecteur. | `domaine`, `infra`. |
| `infra` | Ressources, horloge, aléa, reporting, notification. | Bibliothèque standard. |
| `reseaux` | Caractéristiques, MLP, mutation, stockage, entraînement. | `domaine`, `strategies`, `actions`, NumPy. |
| `selection` | Glouton, aléatoire, epsilon, politique réseau. | `reseaux`, `strategies`. |
| `recompenses` | Variantes de récompense. | `domaine`, `regles`. |
| `evaluation` | Scénarios, métriques, benchmark. | `deroulement`. |
| `composition` | Profil, registre et assemblage par défaut. | Tous les adaptateurs. |
| `ui` | Thème, assets, rendu, clavier, fenêtre Tkinter. | `physique`, `simulateur`. |

Les dépendances autorisées vont du domaine vers l'application, jamais
l'inverse. Un test d'isolation vérifie que `domaine` n'importe ni Pymunk, ni
NumPy, ni Tkinter, et que le moteur n'importe pas le chargeur de robot ni
l'interface au niveau module.

### Façades de compatibilité

| Ancien module | Contenu actuel |
| --- | --- |
| `src/simulateur.py` | Réexporte le domaine, le terrain, le robot, les règles et `Simulation`. |
| `src/match.py` | Réexporte la planification, les actions, les stratégies, le déroulement et les replays. |
| `src/reseau.py` | Réexporte les caractéristiques, le réseau, la politique, la mutation et les métriques ; garde la boucle d'entraînement. |
| `src/interface.py` | Réexporte `create_window` depuis `ui/window.py`. |

## Flux d'exécution

Les noms de fonctions historiques (`play_match`, `choose_plan`, `_route`, etc.) restent ceux des façades. Les flux décrivent donc le comportement observable, pas l'emplacement actuel de chaque symbole.

### 1. Pilotage manuel

```text
python -m src.interface
  -> Simulation()
  -> create_window(simulation)
  -> boucle Tkinter toutes ~16 ms
       -> simulation.advance(now - last_frame)
            -> _step() à PHYSICS_STEP fixe
       -> draw()
            -> lit position, footprint, blocks, suction_cups, movement, score
       -> clic / flèches / Q-D / touches 1-0
            -> simulation.move_to / turn_by / toggle_suction
```

L'interface ne modélise aucune règle. Elle convertit les pixels en millimètres avec `scale` et `border`, puis appelle le moteur. `reset_match()` appelle `simulation.reset()`.

### 2. Match de référence

```text
python -m src.match
  -> play_match(Simulation())
       -> record("Départ")
       -> _play_reference(simulation, record)
            -> choose_plan(simulation, unavailable)
                 -> scoring_zones / construction_score
                 -> _pickup_approach / _approach / _route
            -> _take_pickup(...)
            -> déplacements vers plan.drop
            -> simulation.toggle_suction(cup)
            -> _drop_retreat(...)
            -> répète jusqu'à BUILD_DEADLINE = 90 s
       -> _return_home(simulation, record)
       -> replay = {obstacles, frames, score, score_retour, duree}
  -> save_replay(replay)
```

`choose_plan` est une heuristique gloutonne : elle classe les couples (prise, dépôt) par gain de score divisé par durée estimée. Elle ne consulte pas le réseau.

### 3. Match avec réseau

```text
play_match(simulation, policy=MatchPolicy(network))
  -> batches = True
  -> _play_batches(simulation, record, policy, drop_poses)
       -> pour chaque ventouse libre :
            -> _pickup_approach pour chaque couple (pile, ventouse)
            -> policy.order(...)
                 -> features(simulation, held, choices, pickup=True, pickup_plans=...)
                 -> network.predict(inputs)
                 -> ordre des candidats + exploration éventuelle
            -> _take_pickup(...)
       -> pour les piles tenues :
            -> calcul des placements (gain, durée)
            -> policy.order(..., pickup=False)
            -> place_pile(...)
       -> répète jusqu'à BUILD_DEADLINE
  -> _return_home(...)
```

`MatchPolicy.observe()` mémorise les lignes d'entrée et le score avant l'action pour l'apprentissage. La politique peut aussi renvoyer `None` pour demander le retour ou arrêter le chargement.

### 4. Entraînement

```text
python -m src.reseau train
  -> Network.load(...) éventuel
  -> train_matches(...)
       -> ProcessPoolExecutor
            -> _simulate_match((network, seed, warmup, resume, random_match))
                 -> play_match(Simulation(), policy=MatchPolicy(...), record_frames=False)
                 -> retourne (replay, history)
       -> _train_matches(...)
            -> mute les poses de dépôt
            -> entraîne des copies du réseau sur l'historique du meilleur match
            -> évalue tous les candidats sur la table officielle
            -> garde le meilleur rang (score, -durée)
       -> _validated_replay(...)
  -> network.save(reseau.npz)
  -> save_replay(replay)
```

Le mode `--endless` enchaîne les séries : il repart d'un réseau aléatoire après `PROGRESS_WINDOW = 50` matchs sans amélioration et sauvegarde le meilleur replay de chaque série.

### 5. Lecture d'un replay

```text
python -m src.match replay [numéro]
  -> _trials(REPLAY_DIR)
  -> json.load
  -> Replay(data)
       -> valide les champs et les horodatages
  -> create_window(replay=replay, trial=numéro)
       -> lecture, pause, reprise, curseur
       -> draw() lit replay.frame
```

Le replay ne contient que des données d'affichage : poses, cibles, score, action. Il ne permet pas de reprendre la simulation.

## Cycle de vie du moteur

### `Simulation`

```text
__init__(obstacles, blocks, robot)
  -> reset()
       -> Space Pymunk
       -> bordures statiques
       -> obstacles statiques
       -> robot cinématique à START
       -> piles dynamiques avec joints de freinage
```

`reset()` est le seul point de construction de l'état physique. Toute modification de l'état initial doit passer par là pour rester reproductible.

### Mouvement

```text
arrêt (corps KINEMATIC)
  |
  | move_to(x, y)
  v
translation (corps DYNAMIC) -- move_to(...) --> nouvelle cible
  |
  | arrivée / blocage / _stop()
  v
arrêt

translation + turn_by(angle)
  |-- assez de place : rotation en roulant (_rolling_target)
  |-- pile libre sur la trajectoire : pivot dynamique (_pivot)
  |-- obstacle ou bordure : mouvement bloqué
```

### Ventouses

```text
coupée --toggle_suction(n)--> armée --contact pile--> tient la pile
   ^                                                     |
   +---------------- toggle_suction(n) ------------------+
```

Une ventouse armée reste dans `_armed_cups`. Quand elle capture une pile, elle est retirée de `_armed_cups` et la pile devient un `_Grip` solidaire du châssis. `grip_pile()` est une variante directe, utilisée par les actions automatiques, qui exige l'arrêt du robot.

### Match

```text
0 s ─────────── construction ─────────── 90 s ─── retour ─── 100 s
                 prises / déposes                 salle du trône
```

`play_match` enregistre aussi une frame finale avec le bonus de retour. La durée du replay est le temps simulé atteint, pas nécessairement 100 s.

## Flux de données du réseau

```text
Simulation
  -> blocks, position, heading, held, score, free_wall_slots
  -> features(...) : matrice (N candidats, 97)
       -> Network.predict
       -> ordre des candidats
       -> MatchPolicy.order
       -> action choisie
       -> Simulation
  -> MatchPolicy.observe : (ligne d'entrée, score avant)
  -> _train_matches : cible = (score après - score avant + bonus temps) / 8
```

Le détail des 97 entrées est dans [`domaine-et-regles.md`](domaine-et-regles.md#encodage-du-réseau).

## Points de couplage restants

| Couplage | Conséquence |
| --- | --- |
| `Simulation` utilise encore le terrain et le score officiels par défaut. | Un terrain alternatif doit fournir `TerrainSpec` et les fonctions de score associées. |
| `simulateur.py` et `match.py` restent des façades très larges. | Les nouveaux modules doivent importer les paquets directs, pas les façades. |
| `reseau.py` reste la façade CLI de l'entraînement. | La boucle est dans `reseaux/entrainement.py` et reçoit ses dépendances par `ContexteEntrainement`. |
| `ui/window.py` garde les liaisons Tkinter et la boucle principale. | Le rendu d'état, le clavier, le contrôleur manuel et le contrôleur de replay sont dans `ui/`. |
| Les actions existent en classes `Action`, mais le déroulement appelle encore les fonctions historiques. | `Composition.actions` est rempli et filtré par capacité ; le branchement complet dans `play_match` reste à faire. |
| Un seul robot est simulé. | Les duels et les adversaires ne sont pas encore branchés. |
| Les tours et les portes ne sont pas constructibles avec le modèle actuel de piles de 3 pierres. | Le barème et les classes d'action existent ; il faut modéliser les pierres individuellement. |

## Règles de dépendance recommandées

Pour la suite du projet, viser une dépendance en couches, sans cycle :

```text
domaine (données pures, géométrie, règles)
   ^
   |
physique (Pymunk, corps, collisions)
   ^
   |
planification / actions
   ^
   |
recompenses / selection / reseaux
   ^
   |
application (composition, CLI) et interface
```

Règles concrètes :

- Le domaine ne dépend d'aucun autre paquet du projet et ne dépend pas de Pymunk.
- La physique dépend du domaine, jamais de l'interface, du réseau ou du terrain concret.
- Les actions dépendent du domaine et de la planification, pas de Tkinter ni de NumPy.
- Les réseaux dépendent des caractéristiques et d'un contrat de réseau, pas de `match.py`.
- L'application assemble les implémentations ; c'est le seul endroit qui connaît les identifiants concrets (`robotronik`, `astar`, `mlp`, etc.).
- L'interface dépend de l'application ou du moteur, jamais l'inverse.

Le plan détaillé de migration est dans [`plan-modularisation.md`](plan-modularisation.md).

## Concurrence et erreurs

- **Entraînement** : `ProcessPoolExecutor` exécute `_simulate_match` dans des processus. Les objets injectés doivent être picklables. `Network._for_simulation()` retire la mémoire d'expérience avant l'envoi pour alléger les copies.
- **Sauvegarde** : `save_replay` écrit un fichier temporaire, force `fsync`, réserve un numéro avec un répertoire `.lock`, puis utilise un lien dur sous POSIX ou un `rename` sous Windows.
- **Erreurs attendues** : `ValueError` pour un TOML invalide, une durée négative, une destination non finie, un numéro de ventouse inconnu, un replay invalide. Les CLI convertissent ces erreurs en message et code de sortie 1.
- **Pas de verrou global** : les simulations sont indépendantes ; les caches `lru_cache` sont par processus.

## Points d'extension déjà présents

| Point | Mécanisme actuel |
| --- | --- |
| Forme et ventouses | `RobotRobotronik.toml` + `RobotGeometry` injectable dans `Simulation(robot=...)`. |
| Nombre de ventouses | Liste `[[ventouses]]` et boucles sur `len(simulation.geometry.mounts)`. |
| Politique de sélection | `play_match(policy=...)` accepte un objet avec `order()` et éventuellement `observe()`, `drop_offsets`, `drop_angles`. |
| Poses de dépôt | `drop_offsets` et `drop_angles` de la politique, ou `CENTERED_DROPS` / `ALTERNATE_DROPS`. |
| Obstacles | `Simulation(obstacles=...)` et `replay["obstacles"]`. |
| Blocs initiaux | `Simulation(blocks=...)`. |
| Modèle | `--model` et `Network.load/save`. |
| Replay | Dictionnaire JSON et classe `Replay`. |

Ces points sont utiles mais insuffisants pour l'objectif « chaque morceau interchangeable ». Les contrats manquants sont proposés dans [`plan-modularisation.md`](plan-modularisation.md).
