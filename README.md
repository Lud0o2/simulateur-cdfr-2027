# Simulateur CDFR 2027

Le simulateur sert à comparer des stratégies de match, à entraîner un réseau de neurones contre plusieurs adversaires et à préparer son utilisation sur le robot réel. Le réseau choisira des actions comme « prendre cette pile » ou « déposer une porte dans ce rempart ». Le système de déplacement et les actionneurs exécuteront ces actions.

## Lancer l’étape 1

Depuis le dossier du projet, avec Python 3.12 et Tkinter :

```powershell
python simulateur.py
```

- Clique sur le tapis pour fixer la destination du centre du robot. Les flèches permettent un déplacement de 100 mm.
- Le robot avance en ligne droite en conservant son orientation. Le trait sur le châssis indique l’avant. « Trajet bloqué » s’affiche quand le déplacement s’arrête à un obstacle.
- Appuie sur **Q** pour pivoter à gauche ou sur **D** pour pivoter à droite, par pas de **15°**. Une commande de rotation interrompt le trajet en cours. Si le triangle rencontre un obstacle pendant le pivot, il garde son cap et affiche « Rotation bloquée ».
- En gardant Q, D ou une flèche enfoncée, la commande se répète après **120 ms**, puis toutes les **60 ms**.
- « Replacer au départ » remet le robot dans la salle du trône jaune.

Le châssis provisoire est un triangle équilatéral de 380 mm. Sa vitesse de déplacement est de 1 000 mm/s. Les coordonnées vont de gauche à droite et de haut en bas, en millimètres depuis le coin supérieur gauche du tapis.

Sur la table 2027 vide, les bordures extérieures constituent les obstacles physiques fixes. Les dessins des remparts, tours, douves et ponts sont des zones du tapis. Les piles de pierres de la figure 2 seront ajoutées avec les éléments mobiles à l’étape 2. Le moteur accepte aussi des rectangles fixes pour les scénarios de test.

## Vérifier

```powershell
python -B -m unittest -v
```

Les tests avancent le même moteur qu’utilise la fenêtre. Ils vérifient le déplacement latéral, l’emprise du triangle aux bordures, le contact avec un rectangle fixe, le changement de destination et les collisions pendant la rotation.

## Feuille de route

L’étape 1 est disponible. Les étapes 2 à 8 décrivent le travail prévu et le résultat attendu pour chacune.

### 1. Terrain et déplacement manuel

- Afficher le tapis officiel à l’échelle de la table de 3 000 × 2 000 mm. Piloter le robot holonome de 380 mm de côté au clic et régler son cap avec Q et D. Le centre rejoint la cible en ligne droite en gardant le cap choisi.
- Calculer le premier contact sur tout le trajet, avec l’emprise du triangle. Le moteur peut avancer directement en temps simulé pour jouer sans fenêtre.
- **Validation.** Atteindre une cible dégagée, s’arrêter à une bordure et vérifier qu’un coin du triangle peut toucher un obstacle alors que son centre passe à côté.

### 2. Piles de pierres et prise par le grand côté

- Placer les piles aux emplacements du plan officiel. Suivre le centre et l’orientation des piles dans le simulateur, comme le fera le mât sur la table réelle.
- Approcher une pile par l’un de ses deux grands côtés. Quand le robot est face au bon côté et à portée, il prend les trois pierres d’un coup. La prise a une durée. En transport, la pile est fixe par rapport au châssis et son encombrement compte dans les collisions.
- **Validation.** Prendre une pile à la main, vérifier qu’une approche par le petit côté échoue et qu’un passage franchissable à vide peut bloquer le robot chargé.

### 3. Constructions, choix de pose et score

- Avec une pile, le robot peut déposer les trois pierres empilées en mur, les transformer en porte et déposer l’ensemble en un geste, ou redresser et déposer une tour à la fois en gardant les autres pierres. La transformation et la pose de la porte prennent plus de temps que la pose d’une tour.
- Générer plusieurs positions et orientations par zone de construction. Calculer la pose du châssis nécessaire pour chaque dépôt et vérifier qu’il peut approcher avec sa charge, déposer et repartir. Inclure des poses où une partie des pierres dépasse de la zone.
- Recalculer le score à partir des pierres relâchées, de leur projection, de leurs supports et des bonus de construction au buzzer. Compter une seule porte par château. [Règlement de jeu Eurobot 2027, § E.1 et F](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf)
- **Validation.** Vérifier à la main les cas d’un mur, d’une porte et de tours posées une par une. Tester une pierre partiellement en zone et une pierre encore tenue à la fin.

### 4. Match automatique et replay

- Écrire une stratégie de référence qui choisit une pile accessible et une dépose compatible avec le temps restant. Jouer un match de 100 secondes avec le même moteur qu’en mode manuel.
- Rejouer la partie en affichant les cibles choisies, le résultat des actions, le temps restant et le détail du score. Mesurer le nombre de matchs par seconde sans affichage.
- **Validation.** Relancer le même scénario avec les mêmes commandes et retrouver ses poses, son temps et son score final.

### 5. Premier réseau de neurones

- Fournir au réseau la pose du robot, le temps restant, les positions et orientations connues des pierres, le chargement et les constructions déjà présentes. Lui proposer les actions réalisables, avec une estimation de leur durée et des points qu’elles peuvent rapporter.
- Entraîner le réseau à maximiser le score à 100 secondes. Comparer ses choix à ceux de la stratégie de référence sur les mêmes scénarios.
- **Validation.** Tester deux emplacements de mur qui rapportent autant. L’un est plus rapide à atteindre, l’autre dégage l’accès à une future porte. Vérifier si le réseau adapte son choix au temps restant.

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

L’image locale `tapis_2027.png` permet d’afficher le plan officiel hors ligne. Elle est obtenue à partir de `table_FINALE_1.0.pdf` dans [l’archive du tapis Eurobot 2027](https://www.eurobot.org/wp-content/uploads/2026/10/playmat.zip) publiée sur la [page officielle](https://www.eurobot.org/eurobot-contest/eurobot-2027/). Les dimensions et les bordures suivent le [règlement de jeu, § D.2 et G.1](https://www.eurobot.org/wp-content/uploads/2026/10/Eurobot2027_Rules_FR.pdf).

Pour régénérer l’image avec Poppler après extraction du PDF :

```powershell
pdftoppm -singlefile -scale-to-x 900 -scale-to-y 600 -png table_FINALE_1.0.pdf tapis_2027
```
