# Plan de modularisation

> **Statut : proposition.** Ce document décrit une cible et des étapes. Il ne décrit pas encore le code du dépôt. Les chemins proposés n'existent pas tant que la migration n'est pas réalisée.

## 1. Objectif

Rendre chaque morceau du simulateur interchangeable, avec un critère simple : **pour changer de variante, on ajoute ou on remplace un module, sans modifier les autres**.

Variantes visées :

1. Une fonction de récompense = un module.
2. Une fonction de sélection des candidats = un module.
3. Un robot simulé = un module, avec sa forme, ses ventouses, son algorithme de pathfinding et ses actions possibles.
4. Un algorithme de pathfinding = un module.
5. Un type de réseau = un module.

Contraintes supplémentaires :

- La forme, la position et le nombre de ventouses restent définis dans un fichier TOML.
- Le TOML ajoute le type d'action de construction de chaque ventouse : `mur`, `tour` ou `porte`.
- Une ventouse `tour` peut aussi construire des murs. Une ventouse `porte` peut aussi construire des tours et des murs.
- Le simulateur physique ne doit plus connaître le robot concret.
- Le comportement actuel doit rester le comportement par défaut après migration.

## 2. Résumé exécutif

La cible tient en cinq contrats et une racine de composition :

1. `Planificateur` : un module par algorithme de pathfinding.
2. `Action` : une action par module, filtrée par les capacités du robot.
3. `Recompense` : une variante par module.
4. `Selecteur` : une variante par module ; le réseau est un sélecteur parmi d'autres.
5. `Reseau` + `EncodeurCaracteristiques` : un module par type de réseau et un module d'encodage séparé.

Le `RobotSpec` chargé depuis le TOML porte la géométrie, les ventouses, le mouvement et les capacités de construction. `MoteurPhysique` ne connaît que des corps, des contraintes et un `TerrainSpec`. `app.py` assemble le tout à partir d'un profil.

Ordre recommandé, en quatre lots validables :

| Lot | Phases | Contenu | Risque |
| --- | --- | --- | --- |
| A | 0 à 3 | Baseline, domaine, `RobotSpec` + TOML, frontière physique/robot. | Moyen : touche le moteur. |
| B | 4 et 5 | Planification et actions interchangeables. | Moyen : déplace beaucoup de code de `match.py`. |
| C | 6 et 7 | Récompenses, sélection, réseaux, composition et CLI. | Élevé : touche l'entraînement et les modèles. |
| D | 8 | Tours, portes, base différentielle, duels. | Fonctionnel, après validation de l'architecture. |

Chaque lot doit se terminer par les tests complets et une comparaison au score de référence. Le comportement par défaut ne doit pas changer avant la phase 8.

## 3. État des lieux

| Problème actuel | Effet sur l'objectif |
| --- | --- |
| `simulateur.py` contient le terrain officiel et charge `RobotRobotronik.toml`. | Le moteur dépend du robot et du terrain. |
| `Simulation` code en dur la cinématique holonome, les vitesses, la masse et les seuils de ventouse. | Un autre robot ou une base différentielle impose de modifier le moteur. |
| `match.py` contient à la fois le routeur A*, les actions, la stratégie et le replay. | Changer de pathfinding ou de stratégie impose de toucher au même fichier. |
| `reseau.py` importe des fonctions privées de `match.py`. | La politique, l'encodage et l'apprentissage ne sont pas séparables. |
| `MatchPolicy` lit `network.drop_offsets` et `drop_angles` par `getattr`. | Le contrat entre politique et réseau est implicite. |
| `features()` fixe 97 entrées et 10 piles. | Changer de réseau impose de réécrire l'encodage. |
| Le score de construction est dans `simulateur.py`, la récompense d'entraînement dans `reseau.py`, la sélection de référence dans `match.py`. | Les trois notions sont mélangées. |
| Aucune notion de capacité de construction par ventouse. | Impossible d'exprimer `mur` / `tour` / `porte` dans le TOML. |
| Un seul robot. | Les duels et les adversaires ne sont pas préparés. |

## 4. Principes directeurs

1. **Inversion de dépendance** : les modules de haut niveau dépendent de contrats, pas d'implémentations.
2. **Racine de composition unique** : un seul endroit connaît les identifiants concrets et assemble l'application.
3. **Données immuables** : les descriptions (`RobotSpec`, `TerrainSpec`, `Candidat`, `Trajet`) sont des dataclasses gelées, faciles à mettre en cache et à pickler.
4. **Effets dans les adaptateurs** : la physique, les fichiers, NumPy, Tkinter et les processus restent dans des adaptateurs.
5. **Compatibilité ascendante** : le TOML, les replays et les modèles existants doivent continuer à fonctionner.
6. **Pas de sur-abstraction** : les contrats restent petits ; une variante peut être une simple fonction encapsulée dans une classe.

## 5. Architecture cible

```text
src/
├── app.py                        # CLI et racine de composition
├── composition/
│   ├── config.py                 # lecture/validation du profil TOML
│   ├── registry.py               # identifiant -> fabrique
│   └── defaut.py                 # composition par défaut = comportement actuel
├── domaine/
│   ├── modele.py                 # Pose, Block, SuctionCup, Movement, Candidat
│   ├── geometrie.py              # polygones, SAT, arcs, travel_time
│   ├── terrain.py                # TerrainSpec et requêtes de zones
│   ├── robot.py                  # RobotSpec, Ventouse, Construction, capacités
│   └── score.py                  # règles de score pures
├── terrain/
│   └── officiel_2027.py          # données du terrain officiel
├── robots/
│   └── robotronik/
│       ├── robot.toml            # contour, ventouses, mouvement, planificateur
│       └── plugin.py             # optionnel : planner/actions spécifiques
├── physique/
│   ├── moteur.py                 # MoteurPhysique : corps, contraintes, pas fixe
│   ├── corps.py                  # PileBody, RobotBody
│   └── collisions.py             # requêtes de collision
├── planification/
│   ├── base.py                   # Planificateur, DemandeTrajet, Trajet
│   ├── astar.py                  # routeur actuel
│   ├── direct.py                 # ligne droite, utile aux tests
│   └── grille.py                 # variante future
├── actions/
│   ├── base.py                   # Action, ContexteAction, Candidat
│   ├── prise.py
│   ├── depose.py
│   ├── construction.py           # mur / tour / porte
│   └── retour.py
├── strategies/
│   ├── reference.py              # stratégie gloutonne actuelle
│   ├── lots.py                   # chargement multi-ventouses
│   └── poses.py                  # sites et poses de dépôt
├── recompenses/
│   ├── base.py                   # Recompense
│   ├── construction.py           # score de construction seul
│   ├── construction_temps.py     # score + bonus de temps
│   ├── retour.py                 # bonus de salle du trône
│   └── depots.py                 # récompense par dépose
├── selection/
│   ├── base.py                   # Selecteur
│   ├── glouton.py                # gain / durée
│   ├── aleatoire.py
│   ├── epsilon.py                # décorateur d'exploration
│   └── reseau.py                 # politique actuelle
├── reseaux/
│   ├── base.py                   # Reseau, EncodeurCaracteristiques
│   ├── caracteristiques.py       # encodage 97 entrées
│   ├── mlp.py                    # MLP NumPy actuel
│   ├── lineaire.py               # variante de test
│   ├── entrainement.py           # générations, mutations, workers
│   └── stockage.py               # .npz, métadonnées, migrations
├── replay/
│   ├── format.py                 # validation
│   ├── io.py                     # sauvegarde atomique
│   └── lecture.py                # Replay
├── ui/
│   └── tkinter.py                # interface actuelle
├── simulateur.py                 # façade de compatibilité (phase de transition)
├── match.py                      # façade CLI de compatibilité
├── reseau.py                     # façade CLI de compatibilité
└── interface.py                  # façade CLI de compatibilité
```

Les anciens modules restent des façades tant que les tests et les commandes du README n'ont pas été migrés. Ils peuvent ensuite être supprimés ou réduits à un point d'entrée.

## 6. Contrats

Les contrats sont des `typing.Protocol` avec le minimum de méthodes. Les dataclasses associées vivent dans `src/domaine/`.

```python
Point = tuple[float, float]
Polygon = tuple[Point, ...]
Pose = tuple[float, float, float]  # x, y, cap en radians
```

Le domaine ne doit pas importer Pymunk : les vecteurs sont des paires de flottants, converties en `pymunk.Vec2d` uniquement dans la couche physique.

### 6.1 Robot

```python
class Construction(StrEnum):
    MUR = "mur"
    TOUR = "tour"
    PORTE = "porte"


CAPACITES = {
    Construction.MUR: frozenset({Construction.MUR}),
    Construction.TOUR: frozenset({Construction.TOUR, Construction.MUR}),
    Construction.PORTE: frozenset({Construction.PORTE, Construction.TOUR, Construction.MUR}),
}


@dataclass(frozen=True)
class Ventouse:
    numero: int
    position_mm: Point
    face: Point                      # vecteur unitaire
    construction: Construction = Construction.MUR

    @property
    def capacites(self) -> frozenset[Construction]:
        return CAPACITES[self.construction]


@dataclass(frozen=True)
class Mouvement:
    type: str = "holonome"          # holonome | differentiel
    vitesse_mm_s: float = 1000.0
    acceleration_mm_s2: float = 1000.0
    vitesse_rotation_rad_s: float = radians(360)
    acceleration_rotation_rad_s2: float = radians(1200)
    pas_rotation_rad: float = radians(15)
    rotation_sur_place: bool = True
    deplacement_lateral: bool = True
    masse_kg: float = 5.0


@dataclass(frozen=True)
class RobotSpec:
    id: str
    contour_mm: tuple[Point, ...]
    ventouses: tuple[Ventouse, ...]
    mouvement: Mouvement
    planificateur: str = "astar"

    def supports(self, construction: Construction) -> bool:
        return any(construction in v.capacites for v in self.ventouses)
```

Le chargement TOML valide la géométrie comme aujourd'hui, puis construit `RobotSpec`. Le type `RobotGeometry` actuel devient un alias ou une vue de compatibilité.

Un robot qui a besoin d'un comportement non générique peut fournir `robots/<id>/plugin.py` avec une fabrique de `RobotBody` ou d'`Action`. Le TOML reste la source de vérité pour la géométrie et les capacités.

### 6.2 Moteur physique

```python
class RobotHandle(Protocol):
    id: str
    spec: RobotSpec

    @property
    def pose(self) -> Pose: ...
    @property
    def empreinte(self) -> tuple[Point, ...]: ...
    @property
    def ventouses(self) -> tuple[EtatVentouse, ...]: ...
    def deplacer_vers(self, x: float, y: float) -> None: ...
    def tourner_de(self, angle: float) -> None: ...
    def basculer_ventouse(self, numero: int) -> None: ...
    def pile_tenue(self, numero: int) -> int | None: ...


class MoteurPhysique(Protocol):
    def reset(self) -> None: ...
    def step(self, dt: float) -> None: ...
    def advance(self, seconds: float) -> None: ...
    def robot(self, identifiant: str) -> RobotHandle: ...
    def piles(self) -> tuple[Block, ...]: ...
    def score(self) -> int: ...
    def bonus_retour(self) -> int: ...
```

`MoteurPhysique` ne connaît ni le TOML, ni `START`, ni `START_BLOCKS`, ni `WALL_ZONES`. Il reçoit un `TerrainSpec` et une collection de `RobotSpec` à la construction. L'implémentation Pymunk vit dans `physique/moteur.py` ; une autre implémentation pourrait remplacer Pymunk sans toucher aux actions.

`Simulation` reste une façade :

```python
class Simulation:
    def __init__(self, terrain, robots, *, moteur=None, ...): ...
    def advance(self, seconds): return self._moteur.advance(seconds)
    # propriétés et méthodes historiques déléguées
```

### 6.3 Planification

```python
@dataclass(frozen=True)
class DemandeTrajet:
    depart: Pose
    arrivee: Pose
    empreintes: tuple[tuple[Point, ...], ...]
    obstacles: tuple[Polygon, ...]
    marge: float = 5.0
    angle_vise: float = 0.0
    rotation_en_roulant: bool = True
    limite_s: float = inf


@dataclass(frozen=True)
class Trajet:
    duree_s: float
    avant: tuple[Point, ...]
    rotation_rad: float
    apres: tuple[Point, ...]
    en_roulant: bool = False


class Planificateur(Protocol):
    id: str
    def planifier(self, demande: DemandeTrajet) -> Trajet | None: ...
```

Variantes :

| Module | Identifiant | Description |
| --- | --- | --- |
| `planification/astar.py` | `astar` | Routeur actuel, A* sur points de passage et visibilité. |
| `planification/direct.py` | `direct` | Ligne droite si elle est libre, sinon échec. Utile aux tests et aux robots simples. |
| `planification/grille.py` | `grille` | Grille régulière, à ajouter si les passages étroits le demandent. |
| `planification/visibility.py` | `visibility` | Graphe de visibilité pur, sans points de passage. |

Le planificateur reçoit les empreintes et les obstacles : il ne connaît ni le robot concret ni les actions.

### 6.4 Actions

```python
@dataclass(frozen=True)
class Candidat:
    action: str
    gain: float                      # estimation de la récompense immédiate
    duration_s: float
    robot: str
    payload: Mapping[str, object]


class ContexteAction(Protocol):
    moteur: MoteurPhysique
    robot: RobotHandle
    planificateur: Planificateur
    terrain: TerrainSpec
    temps: float
    deadline: float


class Action(Protocol):
    id: str
    construction: Construction | None

    def candidats(self, ctx: ContexteAction) -> Iterable[Candidat]: ...
    def executer(self, ctx: ContexteAction, candidat: Candidat) -> bool: ...
```

Une action vérifie `ctx.robot.spec.supports(self.construction)` avant de proposer un candidat. `construction=None` pour une action qui n'est pas une construction (retour, déplacement).

Actions prévues :

| Module | Identifiant | Construction |
| --- | --- | --- |
| `actions/prise.py` | `prise` | Aucune. |
| `actions/depose.py` | `depose` | `mur` par défaut, ou la construction du candidat. |
| `actions/construction.py` | `construire_mur`, `construire_tour`, `construire_porte` | Vérifie la capacité de la ventouse. |
| `actions/retour.py` | `retour` | Aucune. |

La priorité de construction est `porte > tour > mur` lorsqu'une même pose peut recevoir plusieurs types. Le type retenu doit être porté par le `Candidat`, pas déduit uniquement de la géométrie.

### 6.5 Récompense

```python
@dataclass(frozen=True)
class Observation:
    robot: Pose
    piles: tuple[Block, ...]
    tenues: Mapping[int, int]        # ventouse -> index de pile
    score: int
    temps: float


@dataclass(frozen=True)
class Transition:
    avant: Observation
    candidat: Candidat
    apres: Observation
    replay: Mapping | None = None


class Recompense(Protocol):
    id: str
    def valeur(self, transition: Transition) -> float: ...
    def rang(self, replay: Mapping) -> tuple[float, float]: ...
```

`valeur` sert à l'apprentissage et à la sélection gloutonne. `rang` sert à comparer deux matchs complets, par exemple `(score, -durée)`.

Variantes :

| Module | Identifiant | `valeur` | `rang` |
| --- | --- | --- | --- |
| `recompenses/construction.py` | `construction` | Delta de score de construction / 8. | `(score, -durée)`. |
| `recompenses/construction_temps.py` | `construction_temps` | Score final (construction + retour) - score avant + bonus de temps. | `(score, -durée)`. |
| `recompenses/retour.py` | `retour` | Delta de score + bonus de retour. | `(score + retour, -durée)`. |
| `recompenses/depots.py` | `depots` | Nombre de déposes réussies. | `(nombre, -durée)`. |
| `recompenses/sparse.py` | `sparse` | `0` sauf à la fin de l'épisode. | `(score, -durée)`. |

Le comportement actuel doit être reproduit par `construction_temps` :

```text
valeur = (score_final_du_replay - score_avant + bonus_temps) / 8
rang   = (score, -durée)
```

`score_final_du_replay` inclut le bonus de retour, comme dans `_train_matches` aujourd'hui. Si l'on veut isoler la construction, utiliser `construction` à la place.

### 6.6 Sélection

```python
class Selecteur(Protocol):
    id: str
    def classer(self, observation: Observation, candidats: Sequence[Candidat]) -> list[Candidat]: ...
    def observer(self, candidat: Candidat | None, valeur: float) -> None: ...


def choisir(selecteur: Selecteur, observation, candidats) -> Candidat | None:
    ordonnes = selecteur.classer(observation, candidats)
    return ordonnes[0] if ordonnes else None
```

Variantes :

| Module | Identifiant | Description |
| --- | --- | --- |
| `selection/glouton.py` | `glouton` | Classe par `reward / duration`, comme `choose_plan`. |
| `selection/reseau.py` | `reseau` | Politique actuelle `MatchPolicy`. |
| `selection/aleatoire.py` | `aleatoire` | Mélange les candidats. Utile aux tests et à l'exploration. |
| `selection/epsilon.py` | `epsilon_glouton` | Décorateur : exploration aléatoire avec probabilité ε. |

La sélection ne contient aucune règle de score : elle utilise les valeurs fournies par la `Recompense` ou par un réseau.

### 6.7 Réseau

```python
class EncodeurCaracteristiques(Protocol):
    schema: str
    def encoder(
        self,
        observation: Observation,
        candidats: Sequence[Candidat],
    ) -> np.ndarray: ...


class Reseau(Protocol):
    id: str
    schema_entree: str

    def predire(self, entrees: np.ndarray) -> np.ndarray: ...
    def entrainer(
        self,
        entrees: np.ndarray,
        cibles: np.ndarray,
        rng: np.random.Generator,
        *,
        learning_rate: float,
    ) -> None: ...
    def sauver(self, chemin: Path) -> None: ...


class FabriqueReseau(Protocol):
    def charger(self, chemin: Path) -> Reseau: ...
```

Variantes :

| Module | Identifiant | Description |
| --- | --- | --- |
| `reseaux/mlp.py` | `mlp` | MLP NumPy actuel `97 → 16 → 8 → 1`, Adam. |
| `reseaux/lineaire.py` | `lineaire` | Régression linéaire, baseline et test de contrat. |
| `reseaux/constante.py` | `constante` | Valeur constante, pour tester la sélection sans apprentissage. |
| Adaptateur externe | `torch_mlp` | Optionnel, si un jour PyTorch est ajouté. |

`EncodeurCaracteristiques` est séparé du réseau : changer de type de réseau ne doit pas imposer de réécrire `features()`, et changer d'encodage doit être explicite via `schema_entree`.

Le fichier `.npz` doit contenir des métadonnées :

```text
network_id       = "mlp"
feature_schema   = "match-2027-v1"
robot_id         = "robotronik"
terrain_id       = "officiel_2027"
reward_id        = "construction_temps"
input_size       = 97
```

`stockage.py` refuse un modèle dont le schéma ou la taille ne correspond pas, avec un message explicite.

### 6.8 Composition

```python
@dataclass(frozen=True)
class Composition:
    robot: RobotSpec
    terrain: TerrainSpec
    planificateur: Planificateur
    actions: tuple[Action, ...]
    recompense: Recompense
    selecteur: Selecteur
    reseau: Reseau | None
    encodeur: EncodeurCaracteristiques | None
```

La racine de composition construit cette structure à partir d'un profil TOML et d'un registre. Aucun autre module ne doit instancier une implémentation concrète.

Exemple de profil :

```toml
[robot]
id = "robotronik"

[terrain]
id = "officiel_2027"

[planification]
id = "astar"

[recompense]
id = "construction_temps"

[selection]
id = "reseau"

[reseau]
id = "mlp"
modele = "reseau.npz"
```

Registre :

```python
FABRIQUES = {
    "robot": {"robotronik": "src.robots.robotronik:construire"},
    "terrain": {"officiel_2027": "src.terrain.officiel_2027:construire"},
    "planification": {
        "astar": "src.planification.astar:construire",
        "direct": "src.planification.direct:construire",
    },
    "recompense": {
        "construction": "src.recompenses.construction:construire",
        "construction_temps": "src.recompenses.construction_temps:construire",
    },
    "selection": {
        "glouton": "src.selection.glouton:construire",
        "reseau": "src.selection.reseau:construire",
        "aleatoire": "src.selection.aleatoire:construire",
    },
    "reseau": {
        "mlp": "src.reseaux.mlp:construire",
        "lineaire": "src.reseaux.lineaire:construire",
    },
}
```

Les imports sont paresseux (`importlib.import_module`) pour éviter de charger Tkinter, NumPy ou Pymunk quand ce n'est pas nécessaire. Un test vérifie que chaque identifiant est unique et résolvable.

## 7. TOML du robot

### 7.1 Schéma proposé

```toml
nom = "Robotronik"
planificateur = "astar"

[mouvement]
type = "holonome"                    # holonome | differentiel
vitesse_mm_s = 1000
acceleration_mm_s2 = 1000
vitesse_rotation_rad_s = 6.283185307179586
acceleration_rotation_rad_s2 = 1200
pas_rotation_rad = 0.2617993877991494
rotation_sur_place = true
deplacement_lateral = true
masse_kg = 5.0

contour_mm = [
    [-219.39310229205776, 0],
    [109.69655114602888, -190],
    [109.69655114602888, 190],
]

[[ventouses]]
position_mm = [109.69655114602888, -100]
angle_deg = 0
construction = "mur"                 # mur | tour | porte

[[ventouses]]
position_mm = [-37.52776749732564, 105]
angle_deg = 120
construction = "mur"
```

### 7.2 Compatibilité

- `construction` absent → `mur`.
- `planificateur` absent → `astar`.
- `[mouvement]` absent → valeurs actuelles.
- `nom` absent → nom du fichier.
- Les champs inconnus restent refusés, mais la liste des champs autorisés est étendue explicitement.
- Les tests existants qui utilisent un TOML minimal doivent continuer à fonctionner.

### 7.3 Règles de capacité

```text
construction = "mur"   -> {mur}
construction = "tour"  -> {tour, mur}
construction = "porte" -> {porte, tour, mur}
```

- Une action de construction vérifie la capacité de la ventouse choisie.
- Un candidat de dépose porte le type de construction visé.
- Une ventouse `tour` peut donc construire un mur si le candidat le demande.
- Une ventouse `porte` peut construire une porte, une tour ou un mur.
- Une ventouse `mur` ne peut jamais construire une tour ou une porte.
- Le score associé aux tours et aux portes reste à préciser avec le règlement Eurobot avant implémentation.

## 8. Catalogue des modules à créer

### Robots

| Module | Contenu |
| --- | --- |
| `robots/robotronik/robot.toml` | Robot actuel, `construction = "mur"`, planner `astar`. |
| `robots/robotronik/plugin.py` | Optionnel ; probablement vide au début. |
| `robots/rectangle_demo/robot.toml` | Robot rectangulaire de test, une ventouse. |
| `robots/differentiel/robot.toml` | Exemple futur avec `type = "differentiel"`. |

### Planification

| Module | Contenu |
| --- | --- |
| `planification/astar.py` | `_route`, `_visibility`, `_approach`, `_turn_envelope`, waypoints. |
| `planification/direct.py` | Trajet direct si libre. |
| `planification/grille.py` | Variante future. |

### Actions

| Module | Contenu |
| --- | --- |
| `actions/prise.py` | `_pickup_approach`, `take_pile`, `_take_pickup`. |
| `actions/depose.py` | `place_pile`, `_drop_approach`, `_drop_retreat`, `_drop_target`. |
| `actions/construction.py` | Vérification des capacités et priorité `porte > tour > mur`. |
| `actions/retour.py` | `_return_home`. |
| `actions/execution.py` | `_follow`, `_stroke_end`, `_action_state`. |

### Récompenses

| Module | Contenu |
| --- | --- |
| `recompenses/construction.py` | Delta de score / 8. |
| `recompenses/construction_temps.py` | Comportement actuel complet. |
| `recompenses/retour.py` | Bonus de salle du trône. |
| `recompenses/depots.py` | Nombre de déposes. |

### Sélection

| Module | Contenu |
| --- | --- |
| `selection/glouton.py` | `choose_plan` sans la partie score. |
| `selection/reseau.py` | `MatchPolicy`. |
| `selection/aleatoire.py` | Mélange reproductible. |
| `selection/epsilon.py` | Décorateur d'exploration. |

### Réseaux

| Module | Contenu |
| --- | --- |
| `reseaux/caracteristiques.py` | `features`, schéma `match-2027-v1`. |
| `reseaux/mlp.py` | `Network` actuel. |
| `reseaux/lineaire.py` | Baseline. |
| `reseaux/constante.py` | Valeur constante. |
| `reseaux/entrainement.py` | `_train_matches`, `_train_endless`, workers. |
| `reseaux/stockage.py` | `.npz`, métadonnées, migration 96 → 97. |

## 9. Migration par phases

Chaque phase doit laisser le dépôt vert et le comportement par défaut inchangé. Utiliser une branche par phase ou une branche d'étape avec des commits séparés.

### Correspondance actuel → cible

| Symbole actuel | Module cible |
| --- | --- |
| `RobotGeometry`, `load_robot` | `domaine/robot.py`, `robots/loader.py` |
| `START`, `START_BLOCKS`, `WALL_ZONES`, `TOWER_CENTERS`, `THRONE_ROOM` | `terrain/officiel_2027.py` |
| `edges`, `rectangle_corners`, `travel_time`, `polygon_contact`, `swept_contact`, `table_contact`, `arc_hits_edge` | `domaine/geometrie.py` |
| `Movement`, `Block`, `SuctionCup` | `domaine/modele.py` |
| `scoring_zones`, `free_wall_slots`, `construction_score`, `throne_room_score` | `domaine/score.py` |
| `Simulation`, `_Pile`, `_Grip` | `physique/moteur.py`, `physique/corps.py` |
| `_route`, `_visibility`, `_approach`, `_turn_envelope`, `_pivot_clear` | `planification/astar.py` |
| `take_pile`, `place_pile`, `_return_home`, `_follow` | `actions/prise.py`, `actions/depose.py`, `actions/retour.py`, `actions/execution.py` |
| `choose_plan` | `selection/glouton.py` + `strategies/reference.py` |
| `_play_batches` | `strategies/lots.py` |
| `features` | `reseaux/caracteristiques.py` |
| `Network` | `reseaux/mlp.py` |
| `MatchPolicy` | `selection/reseau.py` |
| `_train_matches`, `_train_endless` | `reseaux/entrainement.py` |
| `Replay`, `save_replay`, `_frame` | `replay/` |
| `create_window` | `ui/tkinter.py` |

### Phase 0 — Baseline et garde-fous

**But** : pouvoir prouver qu'une refonte ne change pas le comportement.

Actions :

1. Enregistrer les résultats de référence :
   - `python -B -m src.match` : score, durée, nombre de frames, pose finale ;
   - `python -B -m src.reseau compare` avec un modèle fixe ;
   - un match réseau avec une graine fixe.
2. Ajouter des tests de caractérisation qui figent ces valeurs.
3. Vérifier que `python -B -m unittest discover -s test -v` passe.
4. Ne rien déplacer encore.

Critère de sortie : baseline documentée et reproductible.

### Phase 1 — Extraction du domaine pur

**But** : sortir de `simulateur.py` les données et calculs qui ne dépendent ni de Pymunk ni du robot concret.

Actions :

1. Créer `src/domaine/geometrie.py` : `Point`, `Rectangle`, `edges`, `rectangle_corners`, `travel_time`, `polygon_contact`, `swept_contact`, `contact_axes`, `table_contact`, `arc_hits_edge`, `_polygon_bounds`, `_rotated_vertices`, `_footprint_radius`.
2. Créer `src/domaine/modele.py` : `Movement`, `Block`, `SuctionCup`, `Pose`.
3. Créer `src/terrain/officiel_2027.py` : `TABLE_WIDTH`, `TABLE_HEIGHT`, `START`, `START_BLOCKS`, `WALL_ZONES`, `TOWER_CENTERS`, `TOWER_RADIUS`, `THRONE_ROOM`.
4. Créer `src/domaine/score.py` : `scoring_zones`, `free_wall_slots`, `construction_score`, `throne_room_score`.
5. Dans `simulateur.py`, importer et réexporter ces symboles pour ne pas casser les tests.
6. Ne déplacer aucune logique Pymunk.

Vérification : tests verts, match de référence identique, aucun import de `src.terrain` depuis `src.domaine`.

### Phase 2 — `RobotSpec` et capacités TOML

**But** : décrire entièrement le robot dans le TOML, y compris ses actions.

Actions :

1. Créer `src/domaine/robot.py` avec `Construction`, `CAPACITES`, `Ventouse`, `Mouvement`, `RobotSpec`.
2. Déplacer `RobotGeometry` et `load_robot` vers `src/robots/spec.py` et `src/robots/loader.py`.
3. Étendre le TOML avec `planificateur`, `[mouvement]`, `construction` par ventouse.
4. Ajouter les valeurs par défaut pour la compatibilité.
5. Valider les capacités et refuser un type inconnu.
6. Adapter `Simulation` pour accepter un `RobotSpec`.
7. Ajouter les tests de capacité : `mur`, `tour`, `porte`, type inconnu, TOML minimal.

Critère de sortie : le robot par défaut est identique, et un TOML avec une ventouse `tour` produit `{tour, mur}`.

### Phase 3 — Frontière physique / robot

**But** : le moteur ne connaît plus le robot concret.

Actions :

1. Créer `src/physique/moteur.py` avec `MoteurPhysique`.
2. Créer `src/physique/corps.py` avec `PileBody` et `RobotBody`.
3. Déplacer la construction de l'espace Pymunk, des bordures, des piles et du robot.
4. Le moteur reçoit `TerrainSpec` et `RobotSpec`.
5. Déplacer les constantes de cinématique dans `Mouvement`.
6. `Simulation` devient une façade qui délègue au moteur.
7. Ajouter un test d'isolation : le module moteur ne doit pas importer `src.robots`, `src.interface`, `tkinter` ni `PIL`.

Critère de sortie : match de référence identique, test d'isolation vert.

### Phase 4 — Planification interchangeable

**But** : sortir l'A* de `match.py` et injecter un planificateur.

Actions :

1. Créer `src/planification/base.py` avec `DemandeTrajet`, `Trajet`, `Planificateur`.
2. Déplacer le routeur actuel dans `src/planification/astar.py`.
3. Déplacer les waypoints et les sites de dépôt dans `src/strategies/poses.py`.
4. Créer `src/planification/direct.py`.
5. Adapter les actions pour recevoir `ctx.planificateur`.
6. Ajouter un test de permutation : un match court avec `astar`, un autre avec `direct`.
7. Vérifier que `match.py` ne contient plus `_route`.

Critère de sortie : le planificateur par défaut reproduit les trajets actuels ; `direct` fonctionne sur un scénario simple.

### Phase 5 — Actions et capacités

**But** : une action = un module, et les capacités du robot filtrent les actions.

Actions :

1. Créer `src/actions/base.py` avec `Action`, `Candidat`, `ContexteAction`.
2. Déplacer `take_pile` et ses helpers dans `actions/prise.py`.
3. Déplacer `place_pile` et ses helpers dans `actions/depose.py`.
4. Déplacer `_return_home` dans `actions/retour.py`.
5. Créer `actions/construction.py` avec la priorité `porte > tour > mur`.
6. Créer `src/strategies/reference.py` et `src/strategies/lots.py`.
7. Ajouter les tests de capacité : une ventouse `mur` refuse une tour ; une ventouse `tour` accepte un mur ; une ventouse `porte` accepte les trois.

Critère de sortie : score de référence identique ; les capacités sont testées.

### Phase 6 — Récompenses, sélection et réseaux

**But** : séparer les trois notions et rendre chaque variante remplaçable.

Actions :

1. Créer `src/recompenses/base.py` et les variantes.
2. Créer `src/selection/base.py` et les variantes.
3. Créer `src/reseaux/base.py`, `caracteristiques.py`, `mlp.py`, `lineaire.py`.
4. Déplacer `features`, `Network`, `MatchPolicy` et l'entraînement.
5. Faire de `src/reseau.py` une façade CLI.
6. Ajouter les métadonnées de modèle et les tests de compatibilité.
7. Vérifier que `construction_temps` reproduit la cible actuelle `(Δscore + bonus temps) / 8`.

Critère de sortie : match réseau identique avec le modèle existant ; la matrice de permutation passe.

### Phase 7 — Composition et CLI

**But** : un seul point d'assemblage.

Actions :

1. Créer `src/app.py` et `src/composition/`.
2. Ajouter les profils TOML.
3. Ajouter les commandes :

```bash
python -m src.app match --profil configs/defaut.toml
python -m src.app ui
python -m src.app replay 12
python -m src.app train --matches 100 --resume
python -m src.app compare
```

4. Conserver les anciennes commandes comme façades.
5. Ajouter un test qui résout tous les identifiants du registre.

Critère de sortie : les commandes du README fonctionnent toujours ; les nouvelles commandes aussi.

### Phase 8 — Extensions fonctionnelles

**But** : utiliser la modularité pour ajouter les variantes attendues.

Actions :

1. Implémenter les constructions `tour` et `porte` en s'appuyant sur les capacités.
2. Ajouter le score correspondant après avoir cité la section du règlement Eurobot.
3. Ajouter un `Mouvement` différentiel et un robot correspondant.
4. Ajouter un second robot et les duels (étape 8 du README).
5. Ajouter un planificateur `grille` ou `visibility` si les passages étroits le demandent.
6. Ajouter un réseau `lineaire` comme baseline d'apprentissage.

Ces extensions ne doivent pas modifier `physique/`, `domaine/` ni `composition/` au-delà de l'ajout d'un identifiant au registre.

## 10. Vérification

### 10.1 Tests de caractérisation

Figer, pour le scénario par défaut :

- le score final ;
- la durée simulée ;
- le nombre de frames ;
- la pose finale du robot ;
- les indices des piles déposées ;
- le rang `(score, -durée)` d'un match réseau avec une graine fixe.

Les valeurs exactes peuvent dépendre de la plateforme pour les flottants. Comparer avec une tolérance (`places=5`) pour les poses, et exactement pour les scores et les durées arrondies.

### 10.2 Matrice de permutation

Un test paramétré exécute un match court avec :

- planificateurs : `astar`, `direct` ;
- récompenses : `construction`, `construction_temps` ;
- sélecteurs : `glouton`, `aleatoire` ;
- réseaux : `mlp`, `lineaire`, `constante`.

Chaque combinaison doit produire un replay valide et ne jamais lever d'exception. Le but n'est pas que tous les scores soient bons, mais que toutes les combinaisons soient exécutables.

### 10.3 Matrice de capacités

| Ventouse | Mur | Tour | Porte |
| --- | --- | --- | --- |
| `mur` | autorisé | refusé | refusé |
| `tour` | autorisé | autorisé | refusé |
| `porte` | autorisé | autorisé | autorisé |

Chaque case doit être couverte par un test.

### 10.4 Isolation des dépendances

Test AST ou `sys.modules` :

- `src.physique.moteur` n'importe ni `tkinter`, ni `PIL`, ni `src.ui`, ni `src.robots`.
- `src.domaine` n'importe ni Pymunk, ni NumPy, ni Tkinter.
- `src.actions` n'importe ni Tkinter, ni NumPy.
- `src.reseaux` n'importe pas `src.match`.

### 10.5 Interdiction des imports privés

Un test parcourt les imports et refuse les symboles commençant par `_` entre paquets différents. Les helpers partagés doivent devenir publics ou rester dans le même paquet.

### 10.6 Compatibilité

- Un ancien replay sans métadonnées s'ouvre toujours.
- Un ancien modèle `.npz` à 96 entrées se charge toujours.
- Un TOML sans `construction`, `[mouvement]` ou `planificateur` se charge avec les valeurs par défaut.
- Les commandes `python -m src.match`, `python -m src.reseau` et `python -m src.interface` fonctionnent encore.

### 10.7 Performance

Mesurer avant/après :

- durée processeur d'un match de référence ;
- durée d'un match réseau ;
- durée d'un entraînement court.

Les caches actuels (`_route`, `_visibility`, `scoring_zones`, etc.) doivent rester efficaces. Une indirection de protocole ne doit pas ajouter de copie profonde par pas de physique. Budget indicatif : moins de 20 % de régression sur le match de référence.

## 11. Risques et parades

| Risque | Parade |
| --- | --- |
| Régression silencieuse du comportement | Tests de caractérisation avant chaque phase, comparaison des scores et durées. |
| Tests trop couplés aux symboles privés | Migrer progressivement vers des fonctions publiques ; garder des façades pendant la transition. |
| Caches inefficaces ou clés non hachables | Garder les dataclasses gelées ; ne jamais mettre un corps Pymunk dans une clé de cache. |
| Ralentissement de l'entraînement | Profiler `_route` et `_visibility` avant/après ; garder les caches par processus ; éviter les copies de réseau inutiles. |
| Sérialisation des workers | Tester le pickling de chaque composition ; construire les objets dans le worker si nécessaire. |
| Modèles et replays incompatibles | Métadonnées de schéma, migrations explicites, tests de chargement des anciens fichiers. |
| Imports circulaires | Règle de dépendance en couches et test AST. |
| Sur-abstraction | Un contrat par variante réelle ; pas de framework générique. |
| Duplication des constantes pendant la transition | Réexporter depuis un seul module ; supprimer les doublons à la fin de chaque phase. |

## 12. Décisions à valider

1. **Type de construction** : un seul `construction` par ventouse avec fermeture transitive, ou une liste `constructions = [...]` ? Recommandation : un seul type, car il correspond à la formulation « le type est Tour / Porte » et évite les combinaisons incohérentes.
2. **Mouvement dans le TOML** : mettre les vitesses, accélérations et la masse dans `[mouvement]` du robot, ou dans un profil séparé ? Recommandation : dans le TOML du robot, car ce sont ses caractéristiques physiques.
3. **Rôle du robot dans le pathfinding** : le TOML référence un identifiant de planificateur (`planificateur = "astar"`). Le planificateur reste un module séparé, potentiellement partagé par plusieurs robots. Recommandation : référence par identifiant + plugin optionnel.
4. **Récompense et rang** : un seul contrat `Recompense` avec `valeur` et `rang`, ou deux contrats séparés ? Recommandation : un seul contrat, pour que la variante « récompense » contrôle aussi le classement des matchs.
5. **Réseau** : interface NumPy pure ou support optionnel de PyTorch ? Recommandation : NumPy pour le cœur, adaptateur optionnel pour un autre framework.
6. **Registre** : dictionnaire explicite ou `entry_points` de packaging ? Recommandation : dictionnaire explicite tant que le projet n'est pas packagé.
7. **Score des tours et des portes** : les points et les conditions exactes doivent être cités depuis le règlement Eurobot avant implémentation.
8. **Multi-robot** : préparer `MoteurPhysique` pour plusieurs robots dès la phase 3, ou attendre l'étape 8 du README ? Recommandation : garder `robots: dict[str, RobotHandle]` dès la phase 3, même avec un seul robot, pour éviter une seconde refonte.

## 13. Définition de « terminé »

La modularisation est considérée terminée lorsque :

- ajouter un robot ne demande qu'un dossier `robots/<id>/` et une entrée au registre ;
- ajouter un planificateur ne demande qu'un module `planification/<id>.py` et une entrée au registre ;
- ajouter une récompense ne demande qu'un module `recompenses/<id>.py` ;
- ajouter un sélecteur ne demande qu'un module `selection/<id>.py` ;
- ajouter un réseau ne demande qu'un module `reseaux/<id>.py` et, si nécessaire, un encodeur ;
- le comportement par défaut est identique à la baseline ;
- les tests de permutation, de capacité et d'isolation passent ;
- les anciennes commandes et les anciens modèles/replays fonctionnent ;
- `docs/` reflète l'architecture réelle.
