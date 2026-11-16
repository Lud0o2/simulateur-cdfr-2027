# Simulateur CDFR 2027

Le simulateur compare des stratégies de match et entraîne un réseau de neurones sur la table officielle, sans adversaire. Le réseau choisit les prises, les dépôts et le retour à la salle du trône.

## Neuroévolution et replay en direct

Depuis la racine, avec [les dépendances](#commandes-principales) et [le tapis](#source-du-tapis) installés.

| Action | Commande |
| --- | --- |
| Entraîner en continu et regarder le meilleur réseau en boucle | `python -B -m src.neat live` |
| Entraîner 10 générations en arrière-plan, sans fenêtre | `python -B -m src.neat train --generations 10` |
| Ouvrir le dernier replay solo enregistré dans `replays/` | `python -B -m src.match replay` |
| Rejouer le dernier champion solo avec son schéma NEAT | `python -B -m src.neat replay` |
| Resimuler le génome sauvegardé et vérifier toutes les images du match | `python -B -m src.neat verify` |

La fenêtre s’ouvre en plein écran avec le terrain au centre. À gauche, le panneau affiche la génération, le temps écoulé, le nombre de simulations et leur débit, le record, les courbes de score et de temps des 30 dernières générations et le schéma du champion. À droite, seuls les logs défilent. **Arrêter et sauvegarder** termine les évaluations lancées et sauvegarde le meilleur réseau qualifié. Chaque sauvegarde de champion ajoute un replay numéroté dans `replays/`. Le nom du fichier indique le score maximal et le temps auquel il a été atteint pour la première fois. Le journal donne le numéro à passer à la commande, par exemple `python -B -m src.match replay 12`.

Le replay a ses propres commandes. Il tourne en boucle à ×1 par défaut et charge le nouveau record au prochain départ. L’affichage saute l’attente finale si le terrain et le score sont identiques au buzzer. Le fichier représente le match complet de 100 secondes. Le tracé sur le tapis montre les dernières positions réellement parcourues et la cible du segment en cours. Le mode **Agrandir** affiche toutes les connexions du réseau.

La population par défaut est de 200 individus. L’option `--population` permet de la régler. L’entraînement en direct se poursuit jusqu’à la fermeture par défaut. `--generations 5` fixe un budget et `--generations 0` lance un entraînement continu. `--seed 2027` fixe la graine, tirée au hasard sinon. `--sigma` règle l’amplitude des mutations. Elle vaut 0,28 à l’initialisation par défaut et la base des mutations des descendants vaut 1,5 fois cette valeur, soit 0,42. Chaque mutation de descendant applique un facteur tiré parmi 0,5, 1 et 2.

Pour chaque descendant muté, `--add-node` règle la probabilité d’ajouter un neurone (40 % par défaut) et `--add-link` celle d’ajouter une connexion supplémentaire (50 % par défaut). Les deux tirages sont indépendants. Un descendant peut recevoir les deux ajouts si son graphe le permet.

Chaque session enregistre ses résultats dans `neat/<session>/generations.jsonl`. Dès qu’un réseau se qualifie, `champion.json` conserve le meilleur génome, les paramètres, la graine et la trace du match pour `verify`. Pour choisir une session précise avec `src.neat replay` ou `verify`, indique `--directory "neat/<session>"`. `--from-genome "neat/<session>/champion.json"` initialise une nouvelle population autour d’un champion déjà entraîné.

Le réseau évalue chaque candidat à partir de 20 observations, dont la pose et la charge du robot, le temps et le stock restants, le score courant, la durée estimée, le gain du dépôt et la pose visée. Il choisit la pile et la ventouse, la fin du chargement ou le retour, puis l’ordre et la pose des dépôts. Le planificateur vérifie les trajets. L’expérience utilise 24 poses réparties sur les cinq remparts, y compris des poses décalées déjà essayées avec la stratégie de référence.

Un réseau doit atteindre **65 points** pendant le match de 100 s pour se qualifier au classement. Les qualifiés sont classés par temps au premier passage à 65 points. Le champion de la session est le plus rapide. Les points sont relevés lors des prises et dépôts et sur les pas de déplacement du match de 50 ms. Le bonus de retour est acquis quand le robot termine à l’arrêt dans la salle du trône et y reste jusqu’au buzzer. L’évaluation se fait sans fenêtre dans des processus séparés. Un nouveau record est resimulé pour produire le replay et vérifier son score et son temps.

**10 %** de la génération suivante reçoivent de nouveaux poids tirés autour de l’amorce. Un quart des autres parents est tiré uniformément dans une archive en mémoire qui contient jusqu’à 128 stratégies ayant atteint 65 points. L’archive garde le réseau le plus rapide par suite de prises, de dépôts et de fins de série. Pour chaque autre descendant, trois participants sont tirés au hasard dans la génération et le meilleur devient parent, selon le score puis le temps. Ce tournoi fournit tous les parents quand l’archive est vide. Tous les réseaux de la nouvelle génération sont mutés et évalués. Pour partir d’un résultat connu, indique un champion à 65 points avec `--from-genome`.

Lors d’une mutation, chaque poids de connexion a 1 % de chances d’être remplacé par une valeur tirée uniformément entre −1 et 1. Les mutations de structure inspirées de [Pendulum-NEAT](https://github.com/johnBuffer/Pendulum-NEAT) peuvent scinder ou ajouter des connexions. Le graphe impose un ordre acyclique, avec au plus 12 neurones cachés et 120 connexions.

### Mesures du 16 novembre 2026

| Stratégie sur la table officielle | Générations / évaluations | Record |
| --- | ---: | ---: |
| Construction de référence (`reseau.npz`) | — | 65 points en 67,14 s |
| NEAT sans amorce, 36 individus, graine 2027 | 8 / 267 | 55 points en 60,55 s |
| NEAT amorcé sur les décisions de la référence, 40 individus, graine 411 | 8 / 299 | 65 points en 67,14 s |

La population sans amorce est passée de 50 à 55 points. Son champion dépose les dix piles, puis reste bloqué à l’extérieur de la salle du trône. Le réseau d’amorce a été ajusté aux 22 décisions du match de référence. La série amorcée a égalé le record dès la première génération, avec un débit de 1,65 simulation/s sur sept processus sans fenêtre. Les deux replays ont été reproduits image par image avec `verify`.

Avant la sélection à 65 points, `python -B -m src.neat train --population 200 --generations 2 --seed 2027` sur sept processus est passé de 0,959 simulation/s sur `0d220d6` à 1,064 après mutualisation des calculs de dépôt, mise en cache du score et production du replay en parallèle de la génération suivante. Le lot de 384 évaluations a pris 400,28 s puis 360,85 s, sauvegardes comprises. Les 384 scores et temps sont identiques, ainsi que les 2 327 images des deux replays gagnants.

### Recherche de stratégies à 65 points du 16 novembre 2026

Cinq fenêtres de 300 s, avec 40 individus, sept processus, la graine 2027 et le champion de départ `neat/20261116-045917-ary3rfkl/champion.json` à 65 points en 52,69 s. Chaque ligne ajoute un changement à la précédente. Une nouvelle stratégie diffère du départ et des précédentes par la suite des actions réussies, avec la ventouse, la pile en prise et la pose en dépôt. La mesure inclut les fins de série et utilise l’angle de dépôt modulo 180°.

| Essai | Matchs | Matchs à 65 | Nouvelles stratégies | Nouvelles par minute | Meilleure nouvelle |
| --- | ---: | ---: | ---: | --- | ---: |
| 1. Tirage uniforme | 1 026 | 22 | 3 | 3 / 0 / 0 / 0 / 0 | 55,20 s |
| 2. Tournoi à deux | 935 | 42 | 10 | 8 / 0 / 0 / 2 / 0 | 55,04 s |
| 3. Amplitude de base fixe à 0,28 | 898 | 56 | 15 | 8 / 3 / 0 / 1 / 3 | 55,04 s |
| 4. Archive pour un quart des parents | 921 | 173 | 28 | 8 / 2 / 6 / 5 / 7 | 52,88 s |
| 5. Amplitude des descendants à 0,42 | 875 | 146 | 29 | 9 / 7 / 4 / 3 / 6 | 55,04 s |

L’essai 5 avait été retenu à l’issue de la première série pour son nombre de nouvelles stratégies. Les essais 4 et 5 qualifient au moins un réseau par génération complète et trouvent des solutions pendant chacune des cinq minutes. Le record de départ reste à 52,69 s. Le replay local `python -B -m src.match replay 32` montre la nouvelle stratégie la plus rapide, issue de l’essai 4.

Les essais 6 à 10 reprennent l’essai 5 et modifient chacun un seul réglage. Seuls les matchs dont le résultat est reçu pendant les 300 secondes comptent. L’unicité des stratégies est calculée dans chaque essai, puis sur l’ensemble de la deuxième série.

| Essai | Matchs | Matchs à 65 | Nouvelles stratégies | Nouvelles par minute | Meilleure nouvelle |
| --- | ---: | ---: | ---: | --- | ---: |
| 6. Parents tirés dans l’archive à 50 % | 862 | 153 | 20 | 5 / 4 / 7 / 1 / 3 | 52,88 s |
| 7. Perturbation des poids hors connexions directes des entrées communes aux actions | 809 | 117 | 22 | 9 / 6 / 2 / 1 / 4 | 52,88 s |
| 8. Tournoi à trois | 811 | 158 | 26 | 8 / 5 / 8 / 0 / 5 | 55,04 s |
| 9. Ajout de neurone à 20 % chez les descendants | 803 | 150 | 26 | 9 / 2 / 8 / 3 / 4 | 52,88 s |
| 10. Nouveaux génomes à 20 % | 883 | 121 | 11 | 5 / 2 / 2 / 1 / 1 | 52,88 s |
| **Total des essais 6 à 10** | **4 168** | **699** | **61 après dédoublonnage** | — | **52,88 s** |

La deuxième série atteint 65 points dans 16,8 % des matchs (699 sur 4 168), contre 9,4 % dans la première (439 sur 4 655). Le tournoi à trois de l’essai 8 donne le meilleur taux de la deuxième série (19,5 %) et devient le réglage par défaut. L’essai 5 garde le record de 29 nouvelles stratégies en cinq minutes. La meilleure stratégie de la deuxième série reproduit le replay 32. Les deux séries utilisent une seule graine et le même champion de départ.

### Recherche du meilleur temps à 65 points du 16 novembre 2026

Dix essais de 600 s avec 40 individus et sept processus ont joué 12 818 matchs, dont 4 780 à 65 points. Le premier essai, avec `--sigma 0.12` et la graine 2027, a amélioré le record local de **52,64 s à 51,94 s**. Le réseau a modifié les poses de dépôt des piles 1 et 5 dans le même ordre de prises et de dépôts. Les neuf essais suivants sont repartis de ce champion et ont égalé son temps.

Le replay vérifié se lance avec `python -B -m src.match replay 1`. Le génome local se recharge avec `--from-genome "neat/20261116-161113-an0tngvm/champion.json"`.

Dix essais de recherche directe sur les actions ont ensuite évalué 9 257 variantes, dont 1 493 à 65 points. Le meilleur plan atteint **65 points en 36,025 s** sur la table officielle, retour complet inclus. Il parcourt 13,65 m au lieu de 17,74 m pour la référence. Il approche directement de la prise, limite la course après contact à 2 mm et laisse le trajet suivant dégager le châssis après un dépôt. Trois arrêts permettent de déposer deux piles chacun. Les vitesses, l’accélération et le barème sont conservés.

Le replay local **39** contient le plan, réexécutable dans le moteur physique. Les replays **2 à 39** conservent chaque record de la campagne. Cette tournée est calculée pour la table de départ sans adversaire. Sur les mêmes points d’arrêt, les translations idéales seules demandent déjà 29,33 s. Atteindre 30 s laisse donc peu de marge pour les rotations et les détours.

## Point de reprise depuis `main`

- `codex/opti-take-main` repart du commit `55d5e1a` de `main`. La base fournit le pilotage manuel du robot holonome, la physique des piles, quatre ventouses, la planification A*, les actions de prise et de dépose, les matchs de construction et leur replay. Les 73 tests de cette base passent.
- Le dossier local contient les poids de construction `reseau.npz`, lus avec un compteur de 34 matchs.
- Le 15 novembre 2026, la collecte avec chargement complet a ramené les dix piles en **35,17 s** et **13,62 m** parcourus, en décharges de quatre, quatre et deux piles (replay local `3`). La collecte précédente avait parcouru **14,57 m** en **34,50 s** (replay `2`).

## Feuille de route

| Étape | Statut |
| --- | --- |
| [1. Terrain et déplacement manuel](#1-terrain-et-déplacement-manuel) | Validée |
| [2. Piles de pierres et prise par le grand côté](#2-piles-de-pierres-et-prise-par-le-grand-côté) | Validée |
| [3. Score des piles dans les remparts](#3-score-des-piles-dans-les-remparts) | Validée |
| [4. Match automatique et replay](#4-match-automatique-et-replay) | Validée |
| [5. Actions indépendantes, stratégie multi-piles et entraînement](#5-actions-indépendantes-stratégie-multi-piles-et-entraînement) | **Validée** |
| [6. Plusieurs dépôts par zone](#6-plusieurs-dépôts-par-zone) | **À valider** |
| [7. Construction des murs et des portes](#7-construction-des-murs-et-des-portes) | Prévue |
| [8. Duels contre des robots holonomes et différentiels](#8-duels-contre-des-robots-holonomes-et-différentiels) | Prévue |
| [9. Entraînement et comparaison des stratégies](#9-entraînement-et-comparaison-des-stratégies) | Prévue |
| [10. Passage au robot réel](#10-passage-au-robot-réel) | Prévue |

## Commandes principales

Depuis la racine du dépôt, installe les dépendances Python. Pour le pilotage manuel et les replays, prépare aussi [l’image du tapis](#source-du-tapis).

```powershell
python -m pip install -r requirements.txt
```

**Jouer un match de référence, puis ouvrir son replay**

```powershell
python -B -m src.match
python -B -m src.match replay
```

Indique un numéro après `replay` pour ouvrir un essai précis.

**Voir et resimuler le plan à 65 points en 36,025 s**

```powershell
python -B -m src.match replay 39
python -B -m src.match resimuler 39
```

**Jouer un match de collecte**

```powershell
python -B -m src.match collecte
python -B -m src.match replay
```

Le robot revient avec toutes ses ventouses chargées, sauf à la dernière tournée où il ramène toutes les piles restantes. Pour chaque tournée, il simule plusieurs enchaînements et choisit le plus court parmi ceux qui permettent de charger et de rentrer. La distance mesure le déplacement du centre du robot, aller et retour compris. À l’arrêt et entièrement dans la zone de départ, il vide toutes ses ventouses et repart. Seules les piles ramenées avant 100 s comptent. Le chrono s’arrête à la dernière décharge quand toutes les piles ont été ramenées. La zone de départ et les stocks reprennent le terrain officiel (§ D.3, D.4 et plan G.1). [Règlement Eurobot 2027](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

**Entraîner et jouer le réseau de collecte**

```powershell
python -B -m src.reseau collecte --matches 100
python -B -m src.reseau collecte --jouer
python -B -m src.match replay
```

À chaque prise, le réseau choisit parmi les trois piles libres dont les centres sont les plus proches à vol d’oiseau. La planification vérifie ensuite les trajets de prise. À l’avant comme à l’arrière, les deux ventouses d’une paire prennent deux piles voisines sur la même rangée. La ventouse gauche prend celle du côté gauche du châssis et la droite celle du côté droit. Les deux prises se suivent, dans l’ordre choisi par le réseau. Les ventouses latérales peuvent prendre une pile chacune. Le retour se fait à pleine charge ou avec toutes les piles restantes.

À chaque tournée, le réseau peut utiliser les ventouses arrière **5 et 6** dès que les deux ventouses avant **1 et 4** tiennent chacune une pile.

Le réseau **11 → 8 → 1** classe les prises par trajet A*, durée, rotation, proximité du départ et d’une autre pile, ventouse, chargement, stock restant, temps restant et engagement d’une paire. L’entraînement conserve les deux meilleurs réseaux et teste de nouveaux réseaux aléatoires et des mutations. Il maximise le nombre de piles ramenées, puis minimise le temps jusqu’à la dernière décharge. Le fichier local `reseau-collecte.npz` contient les meilleurs poids. Ajoute `--resume` pour repartir de ces poids.

Chaque lancement de l’entraînement de collecte tire une graine aléatoire et l’affiche. Ajoute `--seed` suivi d’un entier pour retrouver les mêmes tirages. Le replay enregistre la graine de la session.

Le 15 novembre 2026, avec six ventouses et 64 matchs d’entraînement (graine 7), le réseau a ramené **10 piles en 26,88 s** sur la table officielle, en deux décharges de six et quatre piles (replay local `7`). Avec quatre ventouses, il avait ramené les dix piles en **30,82 s** après 100 matchs (replay `6`).

**Lancer un battle royale de construction avec 50 réseaux**

```powershell
python -B -m src.reseau train
```

Pour partir avec un fichier de records distinct, choisis un nom de fichier encore libre avec `python -B -m src.reseau --model reseau-nouveau.npz train`.

**Comparer le réseau sauvegardé à la stratégie de référence**

```powershell
python -B -m src.reseau compare
```

**Entraîner en continu**

```powershell
python -B -m src.reseau train --endless
```

## Lancer le simulateur

Utilise Python 3.12 avec Tkinter. Une fois [les dépendances installées](#commandes-principales) et [le tapis préparé](#source-du-tapis), lance le pilotage manuel.

```powershell
python -B -m src.interface
```

Le pilotage manuel et les replays s’ouvrent en plein écran. Le tapis garde ses proportions et s’agrandit dans l’espace disponible entre les informations et les commandes. Appuie sur **Échap** pour revenir en mode fenêtré ou sur **F11** pour basculer entre les deux modes.

- Clique sur le tapis pour fixer la destination du centre du robot. Les flèches commandent un déplacement de 100 mm. Maintiens deux flèches perpendiculaires pour avancer en diagonale.
- Le robot avance en ligne droite vers la destination. La flèche sur le châssis indique l’avant. « Trajet bloqué » s’affiche quand le déplacement s’arrête à un obstacle.
- Appuie sur **Q** pour tourner à gauche ou sur **D** pour tourner à droite, par pas de **15°**, à l’arrêt ou en roulant. Le robot garde sa destination et vérifie l’encombrement pendant la rotation, piles tenues comprises. En pivotant sur place, il peut écarter une pile libre. « Rotation bloquée » s’affiche si le pivot sur place est bloqué ou si le robot atteint sa destination sans pouvoir finir de tourner.
- En gardant Q, D ou une flèche enfoncée, la commande se répète après **120 ms**, puis toutes les **60 ms**.
- Appuie sur la touche associée à la ventouse ou clique sur son repère numéroté, puis avance vers la pile pour la saisir. Le repère est **gris** quand la ventouse est coupée et **orange** quand elle est active, même si elle tient une pile. Répète la commande pour couper la ventouse et lâcher la pile.
- **Reset match** remet le robot dans la salle du trône jaune, replace les piles, coupe les ventouses et remet le temps à zéro.
- **Score jaune** affiche les points de construction selon la position actuelle des piles libres, qu’elles aient été poussées ou déposées.

Le châssis provisoire défini dans `RobotRobotronik.toml` est un triangle équilatéral de 410 mm équipé de six ventouses. Le moteur utilise le triangle comme contour de collision du châssis et ajoute l’encombrement des piles tenues. Le simulateur accélère et freine à 1 m/s² en translation, avec une vitesse maximale de 1 m/s et une vitesse de rotation de 360°/s. Les coordonnées de la table vont de gauche à droite et de haut en bas, en millimètres depuis le coin supérieur gauche du tapis.

La table contient dix piles de trois pierres aux emplacements officiels. Chaque pile est un corps rigide de 320 × 110 mm vu du dessus. Une poussée centrée la fait glisser. Une poussée décentrée peut aussi la faire pivoter, selon le point de contact. Les piles se repoussent et ralentissent sous l’effet du frottement. Le robot s’arrête si la poussée se retrouve bloquée par une bordure ou un obstacle fixe. Les dessins des remparts, tours, douves et ponts sont imprimés sur le tapis. [Règlement Eurobot 2027, § D.4 et G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

Une ventouse active prend les trois pierres d’un coup si elle arrive à **5 mm** d’un grand côté et que les faces sont alignées à **20°** près. La prise peut être décentrée et ne redresse pas la pile. La pile reste solidaire du châssis : son encombrement compte dans les collisions.

La physique tourne à pas fixe avec Pymunk. Les coefficients de frottement, l’accélération et l’effort de pivot sont provisoires en attendant des mesures sur la table.

## Jouer un match automatique

```powershell
python -B -m src.match
python -B -m src.match replay
```

La première commande joue la stratégie de référence sans ouvrir de fenêtre, avec un maximum de 100 secondes simulées. `replay` lit le dernier essai ou le numéro indiqué.

Pour une prise automatique, la ventouse s’aligne à **25 mm** d’un grand côté. Le robot avance ensuite en ligne droite sur **50 mm maximum**, saisit la pile à portée de la ventouse et poursuit tant que le châssis et les piles tenues ont la place. En multi-piles, le robot peut orienter le châssis en roulant si le trajet laisse assez de place pour la rotation.

Après une dépose automatique, le robot se dégage de **50 mm** dans la direction opposée à la ventouse qui vient de lâcher la pile. Ce recul est limité par les obstacles, les autres piles et le temps restant.

En match automatique, le robot tente de rentrer dans la salle du trône dès la fin de ses actions de construction, au plus tard à 90 s. Il vise une entrée complète et peut se rabattre sur une entrée partielle si ses constructions ou son chargement gênent le passage. Le simulateur termine le match et calcule le bonus à la fin de la tentative de retour. Pour un robot à l’arrêt, le bonus vaut **5 points** si le châssis est partiellement dans la salle ou **10 points** s’il y est entièrement. [Règlement Eurobot 2027, § E.3 et plan G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

Un bip bref signale la fin d’un match automatique, d’une comparaison ou d’un entraînement, après sauvegarde du replay. Pendant un entraînement, il sonne une fois en fin de session.

Le replay affiche les poses enregistrées, la cible du robot, l’action en cours, le temps écoulé, l’heure de fin et le score jaune. Clique ou fais glisser le curseur de la barre en bas pour avancer ou reculer. Le bouton à côté met en pause, reprend la lecture ou relance le replay depuis le début une fois arrivé à la fin.

## Modifier le robot

Édite [`RobotRobotronik.toml`](RobotRobotronik.toml), puis relance le simulateur. Le fichier fournit la géométrie au moteur et à l’interface.

- `contour_mm` contient les sommets du châssis, mesurés depuis son centre de rotation. Suis l’ordre du périmètre dans l’un ou l’autre sens. Le contour doit être convexe, sans sommets alignés, et tenir sur le tapis à la position de départ.
- Chaque bloc `[[ventouses]]` définit une ventouse. `position_mm` donne sa position dans le même repère que le contour. `angle_deg` indique vers où elle pointe. L’axe X positif et l’angle 0° pointent vers l’avant. L’axe Y positif et l’angle 90° pointent vers le bas du tapis lorsque le robot est à son cap initial.
- Ajoute, déplace ou supprime des blocs pour changer l’équipement. Leur ordre donne les numéros. Les touches **1 à 9** pilotent les neuf premières ventouses et **0** pilote la dixième. Toutes sont cliquables. Par défaut, **1** et **4** sont à l’avant, à 100 mm de part et d’autre du milieu de la face. **2** et **3** sont sur les côtés. **5** et **6** pointent vers l’arrière depuis la pointe du triangle, à 60 mm l’une de l’autre.

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

Le 14 novembre 2026, l’essai 1 a terminé à **43 points** avec le robot du dépôt, pour **5,12 s** de calcul sans fenêtre. Deux simulations ont produit les mêmes poses enregistrées, et la lecture Tkinter a été vérifiée. Les prises sont centrées et les détours se limitent à un point de passage par trajet.

### 5. Actions indépendantes, stratégie multi-piles et entraînement

**Validée le 14 novembre 2026.** Les poids utilisés lors de la validation ont donné **50 points en 19,96 s simulées**, dont **10 points au retour**, avec le Robotronik à quatre ventouses sur la table officielle sans adversaire.

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

Chaque action renvoie `True` si elle réussit. Les trajets vérifient l’encombrement avec le chargement complet. Un dépôt de pile doit toucher un rempart et son centre doit être hors des cercles de tour. Si la pose est refusée ou si aucune route n’est trouvée, l’action renvoie `False` avant de déplacer le robot. Si la physique interrompt un trajet en cours, le robot s’arrête et garde les piles qu’il tient. A* relie les points visibles, y compris des points proches des coins des piles et obstacles. Le robot suit directement les segments retenus, avec une marge de 5 mm autour des obstacles. Un segment qui rejoint ou quitte une pose serrée utilise le contour réel pour permettre la prise ou la dépose. Le planificateur vérifie l’espace balayé pendant la rotation en roulant et essaie aussi des pivots à l’arrêt. Après une dépose, le robot recule jusqu’à 50 mm si la place le permet. [Règlement Eurobot 2027, § D.4 et E.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

La prise automatique utilise un seul décalage par ventouse. Une ventouse seule saisit le milieu du grand côté. Si deux ventouses partagent une face, chaque pile est décalée vers l’extérieur pour laisser 10 mm entre deux piles de même taille. Avec le robot actuel et des piles de 320 mm, cela décale chaque pile de 65 mm.

La stratégie multi-piles propose d’abord les prises aux trajets les plus courts en temps, rotation comprise. Sans réseau, entre des prises à 50 ms de la plus rapide, elle privilégie celle qui demande le plus petit changement de cap. Si une prise échoue en cours de déplacement, elle recalcule les autres prises depuis la nouvelle pose.

Elle classe les dépôts par gain immédiat, durée estimée et dégagement du passage central. Le robot peut aborder chaque dépôt dans les deux sens. Les cercles sont réservés aux futures actions de tour. Des piles peuvent rester en charge si les remparts deviennent inaccessibles. [Règlement Eurobot 2027, § D.4 et plan G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

#### Battle royale de construction

Un réseau de construction neuf **97 → 32 → 16 → 1** a **3 681 paramètres**. Il reçoit l’état du match et une action candidate, puis renvoie une valeur qui sert à classer les actions. Une pile est en carrière si elle est libre et si son centre se trouve dans un stock officiel. L’occupation des remparts est divisée par cinq. L’indice 84 donne une distance divisée par la diagonale de la table (environ 3 606 mm). Il utilise le trajet planifié pour une prise et la distance directe jusqu’à la pose du châssis pour classer les dépôts. Le simulateur calcule le trajet complet pour les dépôts essayés. La distance vaut −1 si la prise ou la pose finale est invalide et 0 pour l’action d’arrêt du chargement ou de retour. Les indices 85 à 88 décrivent la pose visée. Les anciens poids à 96 entrées et les réseaux à couches cachées élargies restent rechargeables.

Chaque match repart des dix piles centrées sur leurs zones de stockage sur la table officielle. Le réseau classe jusqu’à huit prises et tous les dépôts proposés, avec les deux sens d’approche de chaque rempart. Dès la première prise, il peut arrêter de charger et commencer à déposer. À vide, il peut choisir de rentrer et de terminer. [Règlement Eurobot 2027, plan G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

1. Le tournoi joue un match avec chacun des **50 réseaux initialisés aléatoirement**.
2. À chaque manche, chaque survivant produit **dix mutants** à partir des mêmes poids et poses de départ. Chaque mutation perturbe les poids ou ajuste une pose de dépôt. Le meilleur des onze reste en lice, classé par points puis par temps. À égalité complète, le parent garde sa place.
3. Le réseau qui obtient la plus petite valeur de **points − temps / 2** est éliminé. Le temps est mesuré en secondes. À valeur égale, le réseau qui a le moins de points sort. Il reste un gagnant après **49 manches**, après **490 essais de mutation** sur sa lignée.

Les matchs sont répartis entre les processus disponibles. Après les matchs de départ, le simulateur rejoue seulement les mutants et réutilise les résultats des parents. Un tournoi complet compte **12 790 matchs de classement**, auxquels s’ajoutent les replays et la validation éventuelle du champion déjà sauvegardé. Les poses évoluent dans un rayon de 200 mm autour du centre du rempart, avec des angles au pas de 10°.

Le fichier `reseau.npz` contient le record global, ses poids et ses poses. Si ce fichier existe, son modèle est rejoué comme référence au début. À chaque nouveau record, le programme affiche `Record` et sauvegarde le modèle ainsi qu’un replay dans `replays/`. La fin affiche le replay du meilleur résultat conservé. Ouvre-le avec `python -B -m src.match replay` ou en ajoutant son numéro.

`train --endless` enchaîne les tournois jusqu’à `Ctrl+C`. Chaque tournoi crée 50 nouveaux réseaux et compare leurs résultats au record global. Chaque lancement tire une graine au hasard et l’affiche. Le replay indique la graine de la session. Utilise `--seed` suivi de la graine affichée pour reproduire les tirages.

`compare` recharge les poids, joue le réseau et la référence de l’étape 4 sur la même table officielle, affiche leurs scores et leurs durées, puis garde le replay du réseau. L’option `--model` avant la sous-commande choisit le fichier de poids.

### 6. Plusieurs dépôts par zone

**À valider.**

- Chaque rempart propose quatre emplacements standard. Le centre, les deux décalages latéraux de 120 mm et un décalage diagonal de 200 mm (−120 mm dans la longueur et −160 mm dans la largeur) sont échantillonnés de −90° à +80° par pas de 10°. Pour chaque rempart, le réseau peut aussi choisir la pose sauvegardée par l’entraînement, y compris hors grille.
- Le simulateur écarte les emplacements occupés ou hors rempart avant de chercher un trajet, puis vérifie l’approche avec tout le chargement. Le robot peut continuer à prendre des piles lorsque chaque rempart contient déjà une pile.
- Une pile supplémentaire dans le même rempart ajoute **3 points**. Le bonus de **5 points** du rempart est compté une seule fois.

Le 14 novembre 2026, trois matchs avec les poids de reprise (un autonome, puis deux avec exploration, graines 7 et 11) ont pris **43,2 s de temps processeur au lieu de 61,5 s**. La recherche réutilise les vérifications de rotation pour une même pose et les mêmes obstacles, et écarte plus tôt les détours trop longs. Les trois replays et leurs historiques de décisions sont identiques avant et après.

Le 14 novembre 2026, la stratégie multi-piles a marqué **62 points en 55,08 s** avec les nouveaux points de passage et la marge de 5 mm. Elle a déposé neuf piles et est rentrée entièrement dans la salle du trône. Le replay 64 avait donné le même score en 56,90 s. Sur la prise de la pile 4, le trajet est passé de 3,18 m à 0,98 m entre les deux essais.

Le 15 novembre 2026, la réutilisation des contours tournés, le filtrage des remparts éloignés et la préparation des entrées communes ont ramené de **8,42 s à 6,52 s** le total des temps processeur médians sur trois scénarios (autonome et exploration avec les graines 7 et 19). Les replays et les décisions concordaient à poids et grille identiques. Avec la grille réduite de **630 à 360 poses standard**, le réseau de cet essai a marqué **59 points en 51,72 s simulées** sur la table officielle, contre **59 points en 51,15 s** avec sept positions par rempart.

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
