# Domaine et règles implémentées

Ce document décrit ce que le code fait réellement. Il ne remplace pas le règlement Eurobot.

## Références réglementaires

| Sujet | Référence utilisée dans le code |
| --- | --- |
| Dimensions et bordures du tapis | Règlement Eurobot 2027, § D.2 et plan G.1 |
| Salle du trône | § D.3 et plan G.1 |
| Piles, remparts, tours | § D.4 et plan G.1 |
| Bonus de retour | § E.3 et plan G.1 |
| Image du tapis | Archive officielle `playmat.zip`, PDF `table_FINALE_1.0.pdf` |

Liens utilisés par le README :

- règlement : <https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf>
- tapis : <https://www.eurobot.org/wp-content/uploads/2026/10/playmat.zip>

Toute modification du terrain, du score ou des actions doit citer la section correspondante et mettre à jour les tests.

## Repère et unités

- Origine : coin supérieur gauche du tapis.
- Axe X : vers la droite, de `0` à `TABLE_WIDTH = 3000` mm.
- Axe Y : vers le bas, de `0` à `TABLE_HEIGHT = 2000` mm.
- Cap `0` : le robot regarde vers +X.
- Angles du moteur : radians, normalisés dans `]-π, π]` par `_set_heading`.
- Angles du TOML et de l'interface : degrés.
- Durées : secondes. La physique avance par pas de `1/480` s.

## Terrain

### Table

La table est un rectangle de `3000 × 2000` mm. Ses quatre bordures sont des segments Pymunk statiques. Les remparts, tours et douves dessinés sur le tapis ne sont pas des obstacles physiques par défaut : `Simulation()` part avec `obstacles=()`. Les tests et certains scénarios peuvent fournir des obstacles rectangulaires fixes.

### Piles initiales

`START_BLOCKS` contient dix rectangles `(left, top, right, bottom)` :

- Huit piles horizontales de `320 × 110` mm :
  - rangée du haut, `y = 0..110`, centres `x = 750, 1250, 1750, 2250` ;
  - rangée du bas, `y = 1890..2000`, mêmes centres.
- Deux piles verticales de `110 × 320` mm :
  - `(1445, 340, 1555, 660)` ;
  - `(1445, 1340, 1555, 1660)`.

Chaque pile est un corps dynamique de masse `PILE_MASS = 0.09` kg. Sa taille physique est `max(largeur, hauteur), min(largeur, hauteur)`, avec un angle de `π/2` pour les piles verticales. Les joints `PivotJoint` et `GearJoint` simulent le freinage du tapis sans ancrer la pile.

### Remparts

`WALL_ZONES` contient cinq rectangles de `400 × 200` mm :

| Indice | Centre | Rotation |
| --- | --- | --- |
| 0 | `(250, 400)` | 0° |
| 1 | `(675, 575)` | +45° |
| 2 | `(850, 1000)` | +90° |
| 3 | `(675, 1425)` | -45° |
| 4 | `(250, 1600)` | 0° |

Une pile marque dans un rempart si son polygone chevauche la zone, avec une tolérance de bord de `2 * CONTACT_GAP`.

### Tours

`TOWER_CENTERS` contient quatre centres : `(500, 400)`, `(850, 750)`, `(850, 1250)`, `(500, 1600)`. Une pile marque dans une tour si son rectangle orienté est à moins de `TOWER_RADIUS = 100` mm du centre.

### Salle du trône

`THRONE_ROOM = (0, 750, 500, 1250)`.

- `10` points si les quatre coins du châssis sont entièrement dans le rectangle.
- `5` points si le châssis chevauche le rectangle sans être entièrement dedans.
- `0` point sinon.

Le bonus n'est calculé que lorsque le robot est à l'arrêt.

## Robot

### Géométrie

Le robot est défini par `RobotRobotronik.toml` :

```toml
nom = "Robotronik"
planificateur = "astar"

contour_mm = [[x1, y1], [x2, y2], [x3, y3], ...]

[mouvement]
type = "holonome"
vitesse_mm_s = 1000
acceleration_mm_s2 = 1000
vitesse_rotation_rad_s = 6.283185307179586
acceleration_rotation_rad_s2 = 1200
pas_rotation_rad = 0.2617993877991494
rotation_sur_place = true
deplacement_lateral = true
masse_kg = 5.0

[[ventouses]]
position_mm = [x, y]
angle_deg = 0
construction = "mur"     # mur | tour | porte
```

- `contour_mm` : contour convexe, ordre du périmètre, au moins trois sommets, sans sommets alignés.
- `position_mm` : position de la ventouse dans le repère du robot.
- `angle_deg` : direction de la ventouse, `0°` vers +X.
- `construction` : capacité de construction de la ventouse, `mur` par défaut.
- `planificateur` : identifiant du planificateur associé, `astar` par défaut.
- `[mouvement]` : vitesses, accélérations, pas de rotation, type de base et masse ; les valeurs absentes reprennent les constantes actuelles.
- Les ventouses sont numérotées de `1` à `N` dans l'ordre des blocs.
- Le contour doit tenir dans la table lorsque le robot est à `START = (250, 1000)`.

Les capacités suivent la fermeture demandée : `mur` ne construit que des murs, `tour` construit des tours et des murs, `porte` construit des portes, des tours et des murs. Le score des tours et des portes reste à préciser avec le règlement.

Le robot par défaut est un triangle équilatéral de 380 mm de côté avec quatre ventouses : deux à l'avant (`y = ±100`) et deux sur les côtés (`angle = ±120°`).

### Corps physique

- À l'arrêt et en translation simple, le robot est un corps `KINEMATIC` : sa vitesse est imposée, il ne subit pas les chocs.
- Pour pousser une pile ou tourner contre une pile, le corps devient `DYNAMIC` avec `ROBOT_MASS = 5.0` kg.
- En pivot, un `PivotJoint` fixe le centre et un moment d'inertie est calculé sur le contour.
- La masse, l'accélération et l'effort de pivot sont **provisoires** en attendant les mesures sur le robot réel.

### Ventouses

Une ventouse peut prendre une pile si :

- la pile n'est pas déjà tenue ;
- la ventouse arrive à moins de `SUCTION_REACH = 5` mm d'un grand côté ;
- les faces sont alignées à moins de `20°` (`SUCTION_ALIGNMENT = cos(20°)`) ;
- la ventouse est du bon côté de la pile.

La prise peut être décentrée et ne redresse pas la pile. La pile tenue devient une forme du corps du robot : son encombrement compte dans les collisions et les rotations. `toggle_suction` coupe une ventouse, relâche la pile ou l'arme. `grip_pile` est la variante automatique, qui exige l'arrêt.

## Déplacements

### `move_to(x, y)`

- Destination finie.
- Le robot avance en ligne droite vers la destination.
- Avant de démarrer, `_static_travel` vérifie la table et les obstacles fixes. Les piles ne sont pas testées ici : la physique les gère.
- Si un obstacle fixe ou une bordure bloque, `Movement.blocked` vaut `True` et le robot s'arrête avant le contact.
- La vitesse suit une accélération trapézoïdale limitée par `SPEED` et `DRIVE_ACCELERATION`.
- Un blocage physique réel (pile coincée, chaîne de piles) est détecté par `_stalled` après `0.25 s` sans progrès.

### `turn_by(angle)`

- L'angle doit être strictement compris entre `-π` et `π`.
- Si le robot roule, la rotation est appliquée en roulant (`_rolling_target`) et la destination est conservée.
- Si la rotation est libre, le cap est changé immédiatement.
- Si une pile libre gêne, le robot passe en dynamique et pousse la pile avec un pivot.
- Si un obstacle fixe ou une bordure gêne, `Movement.blocked` ou `rotation_blocked` est positionné.

### `advance(seconds)`

- Durée finie et positive ou nulle.
- Accumulateur de pas fixe `PHYSICS_STEP`.
- `Simulation.time` avance de la durée demandée, pas seulement du temps effectivement simulé.
- La boucle s'arrête plus tôt si le robot est immobile, aucune ventouse n'est armée et toutes les piles dorment ou sont tenues.
- Les ventouses armées tentent une capture après chaque pas.

## Actions automatiques

| Action | Constantes | Comportement |
| --- | --- | --- |
| Prise | `PICKUP_STANDOFF = 25`, `PICKUP_STROKE = 50`, `PICKUP_GAP = 2` | Aligne la ventouse à 25 mm d'un grand côté, avance de 50 mm au maximum, saisit à 2 mm du contact. |
| Dépose | `DROP_RETREAT = 50` | Amène la pile à la pose cible, vérifie qu'elle touche un rempart, relâche, puis recule de 50 mm. |
| Route | `ROUTE_MARGIN = 5` | A* sur un graphe de visibilité avec points de passage. |
| Retour | `BUILD_DEADLINE = 90`, fin à `99,95 s` | Cherche une pose à 10 points, puis à 5 points. |

La dépose vérifie la pose finale à `1e-7` près en position et en angle. Une dépose qui ne touche pas un rempart est refusée.

## Score

### Score de construction

`construction_score` parcourt les piles **libres** :

- `+3` points pour chaque pile qui marque dans un rempart ou une tour (trois pierres à 1 point) ;
- `+5` points par rempart distinct occupé.

Deux piles dans le même rempart ne comptent qu'un bonus de mur. L'algorithme tente de réaffecter les remparts pour maximiser le nombre de zones distinctes.

Exemples :

- une pile dans un rempart : `3 + 5 = 8` points ;
- deux piles dans le même rempart : `3 + 3 + 5 = 11` points ;
- une pile dans une tour : `3` points, sans bonus de mur ;
- une pile hors zone : `0` point.

### Bonus de retour

Voir [`Salle du trône`](#salle-du-trône). Le score final du replay est `simulation.score + bonus`, calculé après le retour.

### Ce qui n'est pas encore scoré

- Les portes et les tours comme constructions distinctes : le `TODO` de `construction_score` le signale.
- Les pierres isolées ou les constructions partielles autres que les piles complètes.
- Les pénalités, fautes et actions de l'adversaire.

## Chronologie d'un match

```text
0 s ───────────────────── 90 s ──────────────── 100 s
       construction            retour
       prises et déposes       salle du trône
```

- `MATCH_SECONDS = 100`.
- `BUILD_DEADLINE = 90`.
- Le retour vise une arrivée à `99,95 s` au plus tard, pour être à l'arrêt au buzzer.
- Un match peut se terminer avant 100 s si la stratégie n'a plus de candidat.
- `play_match` enregistre toujours une frame finale avec le bonus.

## Format de replay

```json
{
  "obstacles": [[left, top, right, bottom]],
  "frames": [
    {
      "t": 0.0,
      "position": [250.0, 1000.0],
      "heading": 0.0,
      "robot": [[x, y], [x, y], [x, y]],
      "piles": [[[x, y], "..."]],
      "cups": [[[x, y], false]],
      "target": [x, y],
      "blocked": false,
      "score": 0,
      "action": "Départ"
    }
  ],
  "score": 0,
  "score_retour": 0,
  "duree": 100.0,
  "decalages_depot_mm": [[dx, dy]],
  "rotations_depot_rad": [angle]
}
```

Contraintes de `Replay` :

- au moins une frame ;
- `obstacles` : liste de quadruplets finis ;
- chaque frame contient exactement les clés `t`, `position`, `heading`, `robot`, `piles`, `cups`, `target`, `blocked`, `score`, `action` ;
- `t` positif ou nul et strictement croissant ;
- `robot` : polygone d'au moins trois points ;
- `piles` : polygones d'au moins quatre points ;
- `cups` : `[[x, y], bool]` ;
- `blocked` booléen, `score` entier, `action` chaîne ;
- nombre de piles et de ventouses constant sur toutes les frames.

`decalages_depot_mm` et `rotations_depot_rad` sont optionnels. `strategie`, `graine`, `serie`, `ouverture` et `score_reference` peuvent être ajoutés par l'entraînement.

## Encodage du réseau

`features()` produit une matrice `(N, 97)`.

### État global (79 valeurs)

| Nombre | Contenu |
| --- | --- |
| 4 | `x / 3000`, `y / 2000`, `cos(cap)`, `sin(cap)` du robot. |
| 1 | Temps restant / `MATCH_SECONDS`. |
| 1 | Score / 80. |
| 1 | Piles tenues / nombre de ventouses. |
| 7 × 10 | Pour chacune des 10 premières piles : `x / 3000`, `y / 2000`, `cos(angle)`, `sin(angle)`, dans le stock initial, tenue, dans une zone. |
| 2 | Pierres libres / 10, remparts occupés / 5. |

Les piles manquantes sont complétées par des zéros.

### Candidat (18 valeurs)

| Nombre | Contenu |
| --- | --- |
| 1 | `1` pour une prise, `0` pour une dépose. |
| 4 | Pose de la pile : `x / 3000`, `y / 2000`, `cos`, `sin`. |
| 1 | Distance d'action / diagonale de la table, ou `-1` si inconnue. |
| 4 | Pose cible : `x / 3000`, `y / 2000`, `cos`, `sin`. |
| 2 | Position de la ventouse / 380. |
| 2 | Vecteur de face de la ventouse. |
| 1 | Durée estimée / 8. |
| 1 | Gain de score / 8. |
| 2 | Offset cible - robot, normalisé. |

Un candidat `None` (fin de chargement ou retour) est encodé par `-1` suivi de zéros.

## Paramètres provisoires

Ces valeurs ne sont pas issues de mesures sur le robot réel et doivent être confirmées avant de conclure sur une stratégie :

| Paramètre | Valeur actuelle | Nature |
| --- | --- | --- |
| `SPEED` | `1000` mm/s | Provisoire. |
| `DRIVE_ACCELERATION` | `1000` mm/s² | Provisoire. |
| `TURN_SPEED` | `360`°/s | Provisoire. |
| `TURN_ACCELERATION` | `1200` rad/s² | Provisoire. |
| `ROBOT_MASS` | `5.0` kg | Provisoire. |
| `PILE_MASS` | `0.09` kg | Provisoire. |
| Frottements table/robot/piles | `0.7` / `0.6` | Provisoire. |
| `FLOOR_DECELERATION` | `4200` mm/s² | Provisoire. |
| `FLOOR_ANGULAR_DECELERATION` | `10` rad/s² | Provisoire. |
| `SUCTION_REACH` | `5` mm | Provisoire. |
| `SUCTION_ALIGNMENT` | `20°` | Provisoire. |
| `PICKUP_STANDOFF`, `PICKUP_STROKE`, `PICKUP_GAP` | `25`, `50`, `2` mm | Choix de stratégie, à valider sur table. |
| `DROP_RETREAT` | `50` mm | Choix de stratégie. |
| `ROUTE_MARGIN` | `5` mm | Marge de sécurité. |

Les dimensions du terrain, des piles, des remparts, des tours et de la salle du trône viennent du règlement et sont supposées exactes.

## Limites actuelles

- Un seul robot, sans adversaire.
- Pas de base différentielle : le robot est holonome et peut se déplacer dans n'importe quelle direction.
- Les capacités `tour` et `porte` sont décrites dans le TOML, mais les actions de construction correspondantes et leur score ne sont pas encore implémentés.
- Pas de fautes, de contacts interdits ni d'arbitrage.
- Pas d'interface avec le robot réel.
- Pas de rejeu déterministe à partir du replay : le replay ne contient que l'affichage.
