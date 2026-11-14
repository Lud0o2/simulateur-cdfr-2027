# Simulateur CDFR 2027

Le simulateur compare des stratégies de match et entraîne un réseau de neurones sur la table officielle, sans adversaire. Le réseau choisit les prises, les dépôts et le retour à la salle du trône.

## Feuille de route

| Étape | Statut |
| --- | --- |
| [1. Terrain et déplacement manuel](#1-terrain-et-déplacement-manuel) | Validée |
| [2. Piles de pierres et prise par le grand côté](#2-piles-de-pierres-et-prise-par-le-grand-côté) | Validée |
| [3. Score des piles dans les remparts](#3-score-des-piles-dans-les-remparts) | Validée |
| [4. Match automatique et replay](#4-match-automatique-et-replay) | Validée |
| [5. Actions indépendantes, stratégie multi-piles et entraînement](#5-actions-indépendantes-stratégie-multi-piles-et-entraînement) | **Validée** |
| [6. Plusieurs dépôts par zone](#6-plusieurs-dépôts-par-zone) | **À valider sur `codex/etape-6`** |
| [7. Construction des murs et des portes](#7-construction-des-murs-et-des-portes) | Prévue |
| [8. Duels contre des robots holonomes et différentiels](#8-duels-contre-des-robots-holonomes-et-différentiels) | Prévue |
| [9. Entraînement et comparaison des stratégies](#9-entraînement-et-comparaison-des-stratégies) | Prévue |
| [10. Passage au robot réel](#10-passage-au-robot-réel) | Prévue |

## Commandes principales

Depuis la racine du dépôt, installe les dépendances Python. Pour le pilotage manuel et les replays, prépare aussi [l’image du tapis](#source-du-tapis).

```powershell
python -m pip install -r requirements.txt
```

**Ouvrir le replay de référence de l’étape 5**

```powershell
python -B -m src.match replay 49
```

Omet le numéro pour ouvrir le dernier replay enregistré.

**Reprendre les poids sauvegardés et jouer 100 matchs d’entraînement**

```powershell
python -B -m src.reseau train --resume --matches 100
```

Pour entraîner un nouveau réseau dans un fichier distinct, choisis un nom de fichier encore libre avec `python -B -m src.reseau --model reseau-nouveau.npz train`.

**Tester aussi des positions et des angles de dépôt**

```powershell
python -B -m src.reseau train --resume --matches 100 --mutate-deposits
```

## Lancer le simulateur

Utilise Python 3.12 avec Tkinter. Une fois [les dépendances installées](#commandes-principales) et [le tapis préparé](#source-du-tapis), lance le pilotage manuel.

```powershell
python -B -m src.interface
```

Le pilotage manuel et les replays s’ouvrent en plein écran. Le tapis garde ses proportions et s’agrandit dans l’espace disponible entre les informations et les commandes. Appuie sur **Échap** pour revenir en mode fenêtré ou sur **F11** pour basculer entre les deux modes.

- Clique sur le tapis pour fixer la destination du centre du robot. Les flèches permettent un déplacement de 100 mm.
- Le robot avance en ligne droite vers la destination. La flèche sur le châssis indique l’avant. « Trajet bloqué » s’affiche quand le déplacement s’arrête à un obstacle.
- Appuie sur **Q** pour tourner à gauche ou sur **D** pour tourner à droite, par pas de **15°**, à l’arrêt ou en roulant. Le robot garde sa destination et vérifie l’encombrement pendant la rotation, piles tenues comprises. En pivotant sur place, il peut écarter une pile libre. « Rotation bloquée » s’affiche si le pivot sur place est bloqué ou si le robot atteint sa destination sans pouvoir finir de tourner.
- En gardant Q, D ou une flèche enfoncée, la commande se répète après **120 ms**, puis toutes les **60 ms**.
- Appuie sur la touche associée à la ventouse ou clique sur son repère numéroté, puis avance vers la pile pour la saisir. Le repère est **gris** quand la ventouse est coupée et **orange** quand elle est active, même si elle tient une pile. Répète la commande pour couper la ventouse et lâcher la pile.
- **Reset match** remet le robot dans la salle du trône jaune, replace les piles, coupe les ventouses et remet le temps à zéro.
- **Score jaune** affiche les points de construction selon la position actuelle des piles libres, qu’elles aient été poussées ou déposées.

Le châssis provisoire défini dans `RobotRobotronik.toml` est un triangle équilatéral de 380 mm équipé de quatre ventouses. Le moteur utilise le triangle comme contour de collision du châssis et ajoute l’encombrement des piles tenues. Le simulateur accélère et freine à 1 m/s² en translation, avec une vitesse maximale de 1 m/s et une vitesse de rotation de 360°/s. Les coordonnées de la table vont de gauche à droite et de haut en bas, en millimètres depuis le coin supérieur gauche du tapis.

La table contient dix piles de trois pierres aux emplacements officiels. Chaque pile est un corps rigide de 320 × 110 mm vu du dessus. Une poussée centrée la fait glisser. Une poussée décentrée peut aussi la faire pivoter, selon le point de contact. Les piles se repoussent et ralentissent sous l’effet du frottement. Le robot s’arrête si la poussée se retrouve bloquée par une bordure ou un obstacle fixe. Les dessins des remparts, tours, douves et ponts sont imprimés sur le tapis. [Règlement Eurobot 2027, § D.4 et G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

Une ventouse active prend les trois pierres d’un coup si elle arrive à **5 mm** d’un grand côté et que les faces sont alignées à **20°** près. La prise peut être décentrée et ne redresse pas la pile. La pile reste solidaire du châssis : son encombrement compte dans les collisions.

La physique tourne à pas fixe avec Pymunk. Les coefficients de frottement, l’accélération et l’effort de pivot sont provisoires en attendant des mesures sur la table.

## Jouer un match automatique

```powershell
python -B -m src.match
python -B -m src.match compare
python -B -m src.match replay 1
```

La première commande joue la stratégie de référence sans ouvrir de fenêtre, avec un maximum de 100 secondes simulées. `compare` joue la référence et la stratégie multi-piles sur la même table, affiche leurs scores et enregistre le replay de la stratégie multi-piles. `replay` lit le dernier essai ou le numéro indiqué.

Pour une prise automatique, la ventouse s’aligne à **25 mm** d’un grand côté. Le robot avance ensuite en ligne droite sur **50 mm maximum**, saisit la pile à portée de la ventouse et poursuit tant que le châssis et les piles tenues ont la place. En multi-piles, le robot peut orienter le châssis en roulant si le trajet laisse assez de place pour la rotation.

Après une dépose automatique, le robot se dégage de **50 mm** dans la direction opposée à la ventouse qui vient de lâcher la pile. Ce recul est limité par les obstacles, les autres piles et le temps restant.

En match automatique, le robot tente de rentrer dans la salle du trône dès la fin de ses actions de construction, au plus tard à 90 s. Il vise une entrée complète et peut se rabattre sur une entrée partielle si ses constructions ou son chargement gênent le passage. Le simulateur termine le match et calcule le bonus à la fin de la tentative de retour. Pour un robot à l’arrêt, le bonus vaut **5 points** si le châssis est partiellement dans la salle ou **10 points** s’il y est entièrement. [Règlement Eurobot 2027, § E.3 et plan G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

Un bip bref signale la fin d’un match automatique, d’une comparaison ou d’un entraînement, après sauvegarde du replay. Pendant un entraînement, il sonne une fois en fin de session.

Le replay affiche les poses enregistrées, la cible du robot, l’action en cours, le temps écoulé, l’heure de fin et le score jaune. Clique ou fais glisser le curseur de la barre en bas pour avancer ou reculer. Le bouton à côté met en pause, reprend la lecture ou relance le replay depuis le début une fois arrivé à la fin.

## Modifier le robot

Édite [`RobotRobotronik.toml`](RobotRobotronik.toml), puis relance le simulateur. Le fichier fournit la géométrie au moteur et à l’interface.

- `contour_mm` contient les sommets du châssis, mesurés depuis son centre de rotation. Suis l’ordre du périmètre dans l’un ou l’autre sens. Le contour doit être convexe, sans sommets alignés, et tenir sur le tapis à la position de départ.
- Chaque bloc `[[ventouses]]` définit une ventouse. `position_mm` donne sa position dans le même repère que le contour. `angle_deg` indique vers où elle pointe. L’axe X positif et l’angle 0° pointent vers l’avant. L’axe Y positif et l’angle 90° pointent vers le bas du tapis lorsque le robot est à son cap initial.
- Ajoute, déplace ou supprime des blocs pour changer l’équipement. Leur ordre donne les numéros. Les touches **1 à 9** pilotent les neuf premières ventouses et **0** pilote la dixième. Toutes sont cliquables. Par défaut, **1** et **4** sont à l’avant, à 100 mm de part et d’autre du milieu de la face. **2** et **3** sont sur les côtés, chacune sur une face.

## Architecture

- `src/simulateur.py` contient le moteur de simulation, utilisable sans fenêtre.
- `src/interface.py` contient l’interface Tkinter, le pilotage manuel et la lecture des replays.
- `src/match.py` planifie et joue les matchs automatiques, puis enregistre les replays.
- `src/reseau.py` entraîne et évalue le réseau.
- `test/` contient les tests de simulation, de replay et d’entraînement.
- `assets/` contient l’image du tapis.

## Vérifier

```powershell
python -B -m unittest discover -s test -v
```

Les tests avancent le même moteur qu’utilise la fenêtre. Ils vérifient le déplacement, les collisions, les poussées, la prise et le relâchement, le score des piles libres et la réinitialisation.

## Détail des étapes

### 1. Terrain et déplacement manuel

**Validée.**

- Afficher le tapis officiel à l’échelle de la table de 3 000 × 2 000 mm. Piloter le robot holonome de 380 mm de côté au clic et régler son cap avec Q et D. Le centre rejoint la cible en ligne droite en gardant le cap choisi.
- Calculer le premier contact sur tout le trajet, avec l’emprise du triangle. Le moteur peut avancer directement en temps simulé pour jouer sans fenêtre.
- **Validation.** Atteindre une cible dégagée, s’arrêter à une bordure et vérifier qu’un coin du triangle peut toucher un obstacle alors que son centre passe à côté.

### 2. Piles de pierres et prise par le grand côté

**Validée.**

- Placer les piles selon le plan officiel, gérer leur poussée et rétablir leur position au reset. La position et l’orientation sont connues dans le simulateur, comme elles le seront grâce au mât sur la table réelle.
- Approcher une pile par l’un de ses deux grands côtés avec la ventouse correspondante activée. La prise est instantanée pour cet essai. Sa durée sera calibrée sur le robot réel. En transport, la pile est fixe par rapport au châssis et son encombrement compte dans les collisions.
- **Essai manuel.** Sans tourner, place le robot à **Y ≈ 500 mm**, active la ventouse **1** et avance vers la pile centrale. Fais pivoter le robot avec la pile, vérifie que le repère reste orange, puis appuie de nouveau sur **1** pour la lâcher. **Reset match** doit replacer toutes les piles et remettre les repères au gris.

### 3. Score des piles dans les remparts

**Validée.** Le score jaune suit la position actuelle des piles libres. Une pile poussée dans un rempart peut marquer des points et les perd en ressortant. Une pile tenue vaut zéro tant que le robot la tient.

- Une pile libre rapporte **3 points** si sa projection touche un rempart jaune, même partiellement. Chaque zone rectangulaire peut donner un **bonus de 5 points** pour un mur. Une pile ne compte qu’une fois, même si elle touche plusieurs zones. [Règlement de jeu Eurobot 2027, § D.4, E.1, F et G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)
- **Essai manuel.** À partir du départ, avance à **X ≈ 250, Y ≈ 500 mm** sans tourner. Active la ventouse **1** et avance vers la pile centrale. Avec la pile tenue, rejoins **X ≈ 650, Y ≈ 1 000 mm**. Une fois à l’arrêt, relâche la pile et vérifie les **8 points**. Reprends-la pour vérifier le retour à **0**. Une pile relâchée qui touche seulement un cercle de tour rapporte **3 points**.

### 4. Match automatique et replay

**Validée.**

- Après chaque dépôt, comparer les prochaines prises et les positions de dépôt qui rapportent des points. Choisir le meilleur gain estimé par seconde en comptant le pivot, les trajets et le dégagement. Le robot garde le même cap pendant le transport. Il essaie des trajets directs ou avec un point intermédiaire.
- Jouer au plus 100 secondes simulées avec la physique du mode manuel. Le fichier de chaque essai conserve le score, la durée et les poses affichées par le replay.
- **Validation.** Lancer deux fois le même scénario, comparer les poses, le score et la durée finale, puis ouvrir chaque essai par son numéro. Le chronométrage affiché permet de mesurer la cadence sans fenêtre.

Le 14 novembre 2026, l’essai 1 a terminé à **43 points** avec le robot du dépôt, pour **5,12 s** de calcul sans fenêtre. Deux simulations ont produit les mêmes poses enregistrées, et la lecture Tkinter a été vérifiée. Les prises sont centrées et les détours se limitent à un point de passage par trajet. [Replay de l’essai 1](replays/essai-0001.json)

### 5. Actions indépendantes, stratégie multi-piles et entraînement

**Validée le 14 novembre 2026.** Avec les poids de `reseau.npz`, le Robotronik à quatre ventouses reproduit **50 points en 19,96 s simulées**, dont **10 points au retour**, sur la table officielle sans adversaire. [Replay de référence](replays/essai-0049.json)

#### Prise et dépôt

Les deux actions sont disponibles dans `src.match`. Elles utilisent le même moteur que le pilotage manuel et se lancent quand le robot est à l’arrêt. L’indice de pile commence à zéro, la numérotation des ventouses à un. Pour une pose, `x` et `y` donnent le centre de la pile en millimètres et `angle` son orientation sur la table en radians.

```python
from math import pi
from src.match import place_pile, take_pile
from src.simulateur import Simulation

sim = Simulation(blocks=((1445, 440, 1555, 760), (1445, 1040, 1555, 1360)))
assert take_pile(sim, pile=0, cup=1)
assert take_pile(sim, pile=1, cup=4)
assert place_pile(sim, cup=4, x=850, y=1200, angle=pi / 2)
```

Chaque action renvoie `True` si elle réussit. Les trajets vérifient l’encombrement avec le chargement complet. Un dépôt de pile doit toucher un rempart et son centre doit être hors des cercles de tour. Si la pose est refusée ou si aucune route n’est trouvée, l’action renvoie `False` avant de déplacer le robot. Si la physique interrompt un trajet en cours, le robot s’arrête et garde les piles qu’il tient. À cap constant, A* cherche le trajet le plus rapide parmi les points de passage disponibles et peut sauter tout point qu’une liaison directe permet d’éviter. Le robot s’arrête aux points de passage restants. Pour tourner en roulant, le planificateur vérifie l’espace balayé par le châssis et les piles tenues sur le changement de cap demandé. Il peut aussi prévoir un pivot à l’arrêt. À la dépose, il peut prévoir un court dégagement avant de recalculer le trajet. [Règlement Eurobot 2027, § D.4 et E.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

La prise automatique utilise un seul décalage par ventouse. Une ventouse seule saisit le milieu du grand côté. Si deux ventouses partagent une face, chaque pile est décalée vers l’extérieur pour laisser 10 mm entre deux piles de même taille. Avec le robot actuel et des piles de 320 mm, cela décale chaque pile de 65 mm.

La stratégie multi-piles limite chaque chargement aux ventouses libres et au nombre de remparts encore libres. Elle propose d’abord les prises aux trajets les plus courts en temps, rotation comprise. Sans réseau, entre des prises à 50 ms de la plus rapide, elle privilégie celle qui demande le plus petit changement de cap. Si une prise échoue en cours de déplacement, elle recalcule les autres prises depuis la nouvelle pose.

Elle classe les dépôts par gain immédiat, durée estimée et dégagement du passage central. Par défaut, la pile est centrée et alignée sur la longueur du rempart. Le réseau peut utiliser les poses retenues par mutation. Si un trajet échoue, le robot essaie aussi d’aborder ce même dépôt avec un demi-tour de différence. Les cercles sont réservés aux futures actions de tour. Des piles peuvent rester en charge si les remparts deviennent inaccessibles. [Règlement Eurobot 2027, § D.4 et plan G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

#### Entraînement et validation

Le réseau **97 → 16 → 8 → 1** a **1 713 paramètres**. Il reçoit l’état du match et une action candidate, puis estime les points que cette action et la suite du match peuvent rapporter. Une pile est en carrière si elle est libre et si son centre se trouve dans un stock officiel. L’occupation des remparts est divisée par cinq. L’indice 84 donne la longueur du trajet planifié, détours A* compris, divisée par la diagonale de la table (environ 3 606 mm). Les indices 85 à 88 décrivent la pose visée. La distance couvre la prise ou le trajet jusqu’à la dépose. Elle vaut −1 sans trajet de prise ou de dépôt et 0 pour l’action d’arrêt du chargement ou de retour. Les poids à 96 entrées restent rechargeables. Les nouvelles entrées de stock et de distance commencent avec des poids nuls et sont apprises lors des entraînements suivants.

L’entraînement alterne la stratégie multi-piles, l’exploration et les décisions du réseau. Chaque match démarre avec les dix piles centrées sur leurs zones de stockage, aux emplacements du plan G.1. La graine fait varier les décisions explorées. Le réseau classe jusqu’à huit prises et tous les dépôts proposés, avec les deux sens d’approche de chaque rempart. Dès la première prise, il peut arrêter de charger et commencer à déposer. À vide, il peut choisir de rentrer et de terminer. Pour chaque décision, il apprend les points marqués d’ici la fin du match, retour compris. Les points tardifs ont le même poids que les points immédiats. Finir plus vite ajoute au plus l’équivalent d’un demi-point aux cibles d’apprentissage. [Règlement Eurobot 2027, plan G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

```powershell
python -B -m src.reseau compare
```

`train --resume` repart des poids de `reseau.npz` et affiche le chemin du fichier chargé. Lors des matchs guidés en reprise, le réseau explore une autre action dans 30 % des décisions, parmi les choix du deuxième au cinquième rang. Quand le robot est chargé, une place est réservée à la prise la plus rapide vers une autre pile. Le taux d’apprentissage est de 0,005 en reprise et de 0,01 avec une nouvelle initialisation.

Un match d’entraînement sur huit choisit ses actions au hasard parmi toutes les prises accessibles et tous les dépôts proposés. Il poursuit la construction tant qu’une prise reste accessible à vide. Pour chaque ouverture, définie par les deux premières piles prises, l’entraînement conserve les décisions et les récompenses du meilleur match qui a marqué des points de construction. Les candidats réutilisent aussi des exemples tirés de cette mémoire. `reseau.npz` conserve cette mémoire et le nombre de matchs joués, même quand le meilleur réseau reste le même. Avec `--resume`, les tirages et la fréquence des matchs aléatoires tiennent compte des sessions précédentes.

Avec `--mutate-deposits`, chaque candidat modifie la position ou l’angle d’un seul dépôt. Le centre de la pile peut se trouver jusqu’à 200 mm du centre du rempart. L’angle peut varier par pas de 15° jusqu’à la position perpendiculaire au rempart. Le simulateur vérifie le trajet, les collisions et la zone. Un match autonome valide chaque mutation. Seule une amélioration du score, ou du temps à score égal, permet de conserver la mutation. Le fichier `reseau.npz` contient les poids et les poses retenues, avec `drop_offsets` pour les décalages en millimètres et `drop_angles` pour les rotations en radians. À la reprise ou en match, le réseau retrouve ces mêmes poses. Un ancien fichier sans `drop_angles` garde les piles alignées sur leur rempart.

L’entraînement choisit automatiquement le nombre de processus selon les cœurs logiques disponibles et le nombre de matchs demandé. Les matchs d’une génération partent du meilleur réseau validé jusque-là. Un candidat privilégie la meilleure trajectoire explorée du lot et les autres apprennent sur l’historique des matchs. Les candidats jouent chacun un match de validation sans exploration sur la table officielle, en parallèle. Le score prime, puis le temps simulé le plus court à score égal. À égalité complète, le meilleur réseau déjà validé garde sa place. Le gagnant sert de point de départ à la génération suivante.

Pour chaque match exploré, le journal affiche le score, la durée simulée et le nombre de décisions retenues, y compris le choix de rentrer. À la fin de chaque génération, il affiche le meilleur résultat sans exploration et le record avec exploration.

Le 14 novembre 2026, un essai de **24 matchs d’entraînement** avec la graine 8 a pris **17,9 s** sur 8 processus, dont **11,9 s** pour les matchs d’entraînement. La validation finale a joué 6 matchs, contre 75 avec la validation après chaque mise à jour. Les poids d’origine ont été conservés lors de cet essai.

Le 14 novembre 2026, la réutilisation des poses et des zones de score a ramené le temps médian pour **24 simulations à poids fixes** de **22,76 s à 17,09 s** sur 8 processus déjà lancés. Les deux versions ont produit les mêmes replays et les mêmes données d’entraînement. Cette mesure exclut la mise à jour des poids et la validation finale.

Sans `--seed`, l’entraînement utilise la graine `7`. Passe `--seed` avec un autre entier pour changer la suite des tirages. La graine utilisée figure dans le replay.

À la fin de chaque entraînement, `train` sauvegarde le meilleur match avec exploration et le meilleur match autonome de la session. Si un réseau déjà sur disque conserve la première place, son match est aussi sauvegardé et affiché sous « Réseau autonome sauvegardé ». Chaque replay a sa commande d’ouverture. Le dernier replay correspond au réseau dans `reseau.npz`.

Le 14 novembre 2026, une reprise de 24 matchs avec mutations et validations sur huit processus a pris **19,91 s au lieu de 37,02 s**, à poids de départ et graines identiques. Le moteur réutilise les contours des piles tant que leur pose reste la même et écarte les détours dont la durée minimale dépasse la meilleure route connue. L’entraînement enregistre les décisions puis calcule les images en rejouant le seul match avec exploration retenu. Les replays sauvegardés et le modèle final étaient identiques dans cet essai.

`compare` recharge les poids, joue le réseau sans exploration et la référence de l’étape 4 sur la même table officielle, affiche leurs scores et leurs durées, puis garde le replay du réseau. L’option `--model` avant la sous-commande choisit le fichier de poids.

La commande `python -B -m src.reseau imitate NUMERO` reprend les poids sauvegardés et apprend d’un replay annoté qui utilise la configuration actuelle du robot. Le replay contient les listes `prises` et `depots`, ainsi que `tournees` pour indiquer le nombre de piles chargées à chaque tournée lorsqu’il y en a plusieurs. La démonstration reprend les décalages et rotations du replay s’il contient `decalages_depot_mm` et `rotations_depot_rad`. L’entraînement rejoue les actions de la démonstration pour apprendre à les classer parmi les choix possibles. La validation lance ensuite un match où le réseau choisit seul. La commande garde le replay de ce match et remplace les poids si le résultat valide progresse en score, puis en durée à score égal.

### 6. Plusieurs dépôts par zone

**À valider.** La branche [`codex/etape-6`](https://github.com/Lud0o2/simulateur-cdfr-2027/tree/codex/etape-6) propose plusieurs emplacements et orientations par rempart. Le robot peut continuer à poser dans une zone déjà occupée si la pose et l’approche restent possibles.

Sur cette branche, `python -B -m src.match replay 64` montre **62 points en 56,90 s simulées**, avec neuf dépôts et un retour complet. [Replay de référence de l’étape 6](https://github.com/Lud0o2/simulateur-cdfr-2027/blob/a96507546bafa3df11a04a0927c39168644152dc/replays/essai-0064.json)

- **À vérifier pour la validation.** Déposer au moins deux piles dans une même zone à des emplacements distincts. Vérifier le comptage des pierres, le passage réservé et le retour du robot.

### 7. Construction des murs et des portes

**Prévue après l’étape 6.**

- Ajouter les actions de construction des murs et des portes à partir des placements disponibles. Vérifier les conditions de construction et les points associés selon le règlement.

### 8. Duels contre des robots holonomes et différentiels

- Faire circuler les deux robots sur la même table et la même horloge. L’holonome peut se déplacer latéralement. Le différentiel doit suivre une trajectoire compatible avec ses roues et s’aligner pour la prise.
- Comparer les bases à stratégie et gabarit identiques, puis faire varier les formes, les vitesses et les stratégies adverses. Déclencher un nouveau choix quand une pile est prise ou qu’un passage devient bloqué.
- **Validation.** Rejouer une course vers la même pile contre chaque type de base. Si l’adversaire arrive le premier, vérifier que notre robot peut changer de cible.

### 9. Entraînement et comparaison des stratégies

- Entraîner contre plusieurs adversaires programmés, puis contre d’anciennes versions du réseau. Varier les ouvertures, les vitesses et les départs.
- Ajouter les autres actions de jeu retenues par l’équipe pour refléter son programme de match.
- **Validation.** Comparer le score moyen, les mauvais résultats et le taux de victoire sur des scénarios réservés à l’évaluation.

### 10. Passage au robot réel

- Alimenter la stratégie avec les positions et orientations remontées par le mât. Faire exécuter les actions choisies par le robot.
- Mesurer sur table les durées de roulage, d’alignement, de prise et de dépose. Utiliser ces mesures pour calibrer le simulateur et réévaluer les stratégies.
- **Validation.** Enchaîner une prise et une dépose sur la table réelle, puis vérifier la réaction à un passage bloqué et l’arrêt à la fin du match.

## Source du tapis

L’image du tapis se prépare localement pour le pilotage manuel et la lecture des replays. Télécharge l’archive officielle et extrais `table_FINALE_1.0.pdf` à la racine du projet. Les dimensions et les bordures suivent le règlement de jeu, § D.2 et G.1. [Archive du tapis Eurobot 2027](https://www.eurobot.org/wp-content/uploads/2026/10/playmat.zip) · [Règlement de jeu](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

Utilise Poppler pour générer `assets/tapis_2027.png` après extraction du PDF.

```powershell
New-Item -ItemType Directory -Force assets | Out-Null
pdftoppm -singlefile -scale-to-x 900 -scale-to-y 600 -png table_FINALE_1.0.pdf assets/tapis_2027
```
