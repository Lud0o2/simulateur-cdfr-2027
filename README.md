# Simulateur CDFR 2027

Le simulateur sert à comparer des stratégies de match, à entraîner un réseau de neurones contre plusieurs adversaires et à préparer son utilisation sur le robot réel. Le réseau choisira des actions comme « prendre cette pile » ou « déposer une porte dans ce rempart ». Le système de déplacement et les actionneurs exécuteront ces actions.

## Lancer le simulateur

Depuis le dossier du projet, avec Python 3.12 et Tkinter. Pour un nouveau clone, prépare d’abord [l’image du tapis](#source-du-tapis).

```powershell
python -m pip install -r requirements.txt
python -B -m src.interface
```

Le pilotage manuel et les replays s’ouvrent en plein écran. Le tapis garde ses proportions et s’agrandit dans l’espace disponible entre les informations et les commandes. Appuie sur **Échap** pour revenir en mode fenêtré ou sur **F11** pour basculer entre les deux modes.

- Clique sur le tapis pour fixer la destination du centre du robot. Les flèches permettent un déplacement de 100 mm.
- Le robot avance en ligne droite en conservant son orientation. La flèche sur le châssis indique l’avant. « Trajet bloqué » s’affiche quand le déplacement s’arrête à un obstacle.
- Appuie sur **Q** pour pivoter à gauche ou sur **D** pour pivoter à droite, par pas de **15°**. Une commande de rotation interrompt le trajet en cours. Le robot peut écarter une pile libre en pivotant. Si une bordure, un obstacle fixe ou une pile coincée bloque le pivot, le simulateur affiche « Rotation bloquée ».
- En gardant Q, D ou une flèche enfoncée, la commande se répète après **120 ms**, puis toutes les **60 ms**.
- Appuie sur la touche associée à la ventouse ou clique sur son repère numéroté. Le repère est **gris** quand la ventouse est coupée et **orange** quand elle est active, même si elle tient une pile. Répète la commande pour couper la ventouse et lâcher la pile.
- **Reset match** remet le robot dans la salle du trône jaune, replace les piles, coupe les ventouses et remet le temps à zéro.
- **Score jaune** affiche les points de construction selon la position actuelle des piles libres, qu’elles aient été poussées ou déposées.

Le châssis provisoire défini dans `RobotRobotronik.toml` est un triangle équilatéral de 380 mm avec deux ventouses à l’avant et une sur chaque autre face. L’ordre des ventouses dans le fichier donne leurs numéros. La vitesse du robot est de 1 000 mm/s. Les coordonnées de la table vont de gauche à droite et de haut en bas, en millimètres depuis le coin supérieur gauche du tapis.

La table contient dix piles de trois pierres aux emplacements officiels. Chaque pile est un corps rigide de 320 × 110 mm vu du dessus. Une poussée centrée la fait glisser. Une poussée décentrée peut aussi la faire pivoter, selon le point de contact. Les piles se repoussent et ralentissent sous l’effet du frottement. Le robot s’arrête si la poussée se retrouve bloquée par une bordure ou un obstacle fixe. Les dessins des remparts, tours, douves et ponts sont imprimés sur le tapis. [Règlement Eurobot 2027, § D.4 et G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)

Une ventouse active prend les trois pierres d’un coup si elle arrive à **5 mm** d’un grand côté et que les faces sont alignées à **20°** près. La prise peut être décentrée et ne redresse pas la pile. La pile reste solidaire du châssis : son encombrement compte dans les collisions.

La physique tourne à pas fixe avec Pymunk. Les coefficients de frottement et l’effort du robot, en poussée et en pivot, sont provisoires en attendant des mesures sur la table.

## Jouer un match automatique

```powershell
python -B -m src.match
python -B -m src.match replay
python -B -m src.match replay 1
```

La première commande calcule un match de 100 secondes sans ouvrir de fenêtre. Elle affiche le score, le temps de calcul et le chemin du fichier numéroté dans `replays/` (`essai-0001.json`, `essai-0002.json`, etc.). La deuxième commande lit le dernier essai, la troisième ouvre un numéro précis.

Le replay affiche les poses enregistrées, la cible du robot, l’action en cours, le temps restant et le score jaune. Clique ou fais glisser le curseur de la barre en bas pour avancer ou reculer. Le bouton à côté met en pause, reprend la lecture ou relance le replay depuis le début une fois arrivé à la fin.

## Modifier le robot

Édite [`RobotRobotronik.toml`](RobotRobotronik.toml), puis relance le simulateur. Le fichier fournit la géométrie au moteur et à l’interface.

- `contour_mm` contient les sommets du châssis, mesurés depuis son centre de rotation. Suis l’ordre du périmètre dans l’un ou l’autre sens. Le contour doit être convexe, sans sommets alignés, et tenir sur le tapis à la position de départ.
- Chaque bloc `[[ventouses]]` définit une ventouse. `position_mm` donne sa position dans le même repère que le contour. `angle_deg` indique vers où elle pointe. L’axe X positif et l’angle 0° pointent vers l’avant. L’axe Y positif et l’angle 90° pointent vers le bas du tapis lorsque le robot est à son cap initial.
- Ajoute, déplace ou supprime des blocs pour changer l’équipement. Leur ordre donne les numéros. Les touches **1 à 9** pilotent les neuf premières ventouses et **0** pilote la dixième. Toutes sont cliquables. Par défaut, **1** et **4** sont à l’avant, à 100 mm de part et d’autre du milieu de la face. **2** et **3** sont sur les côtés.

## Architecture

- `src/simulateur.py` contient le moteur de simulation, utilisable sans fenêtre.
- `src/interface.py` contient l’interface Tkinter et le pilotage manuel.
- `test/` contient les tests du moteur.
- `assets/` contient l’image du tapis.

## Vérifier

```powershell
python -B -m unittest discover -s test -v
```

Les tests avancent le même moteur qu’utilise la fenêtre. Ils vérifient le déplacement, les collisions, les poussées, la prise et le relâchement, le score des piles libres et la réinitialisation.

## Feuille de route

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
- **TODO.** Ajouter les portes et les tours lorsque ces constructions seront simulées.

### 4. Match automatique et replay

**Validée.**

- Après chaque dépôt, comparer les prochaines prises et les positions de dépôt qui rapportent des points. Choisir le meilleur gain estimé par seconde en comptant le pivot, les trajets et le dégagement. Le robot garde le même cap pendant le transport. Il essaie des trajets directs ou avec un point intermédiaire.
- Jouer 100 secondes simulées avec la physique du mode manuel. Le fichier de chaque essai conserve le score et les poses affichées par le replay.
- **Validation.** Lancer deux fois le même scénario, comparer les poses et le score à 100 secondes, puis ouvrir chaque essai par son numéro. Le chronométrage affiché permet de mesurer la cadence sans fenêtre.

Le 14 novembre 2026, l’essai 1 a terminé à **43 points** avec le robot du dépôt, pour **5,12 s** de calcul sans fenêtre. Deux simulations ont produit les mêmes poses enregistrées, et la lecture Tkinter a été vérifiée. Les prises sont centrées et les détours se limitent à un point de passage par trajet. [Replay de l’essai 1](replays/essai-0001.json)

### 5. Premier réseau de neurones

**À concevoir.** Le réseau devra pouvoir enchaîner les prises jusqu’à la capacité réelle du robot, choisir quelle pile relâcher et proposer un dépôt par ses coordonnées et son orientation. Le moteur devra valider la pose et le trajet avec toutes les piles transportées. La comparaison utilisera le score à 100 secondes sur les mêmes scénarios que la stratégie de référence.

### 6. Duels contre des robots holonomes et différentiels

- Faire circuler les deux robots sur la même table et la même horloge. L’holonome peut se déplacer latéralement. Le différentiel doit suivre une trajectoire compatible avec ses roues et s’aligner pour la prise.
- Comparer les bases à stratégie et gabarit identiques, puis faire varier les formes, les vitesses et les stratégies adverses. Déclencher un nouveau choix quand une pile est prise ou qu’un passage devient bloqué.
- **Validation.** Rejouer une course vers la même pile contre chaque type de base. Si l’adversaire arrive le premier, vérifier que notre robot peut changer de cible.

### 7. Entraînement et comparaison des stratégies

- Entraîner contre plusieurs adversaires programmés, puis contre d’anciennes versions du réseau. Varier les ouvertures, les vitesses et les départs.
- Ajouter le retour au château et les autres actions de jeu retenues par l’équipe pour refléter son programme de match.
- **Validation.** Comparer le score moyen, les mauvais résultats et le taux de victoire sur des scénarios réservés à l’évaluation.

### 8. Passage au robot réel

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
