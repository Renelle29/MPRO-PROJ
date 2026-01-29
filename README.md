# MPRO-PROJ
Projet d'optimisation robuste du MPRO de Ronan Leprat et Aurélien Arnoux

Le fichier PROJ contient notre modélisation théorique du problème.

## TODO

- [x] - Implémenter méthode statique.
- [x] - Implémenter méthode robuste par dualisation.
- [x] - Implémenter méthode robuste par plans coupants.
- [ ] - Améliorations plans coupants:
1. Initialisation des ensembles d'incertitudes.
2. Eviter de relancer l'optim de zéro pour le problème principal.
- [ ] - Implémenter méthode robuste par branch-and-cut.
- [ ] - Implémenter une heuristique avec garantie de performances.
- [ ] - Tableau comparatif des performances de chacune des méthodes sur les instances proposées.
- [ ] - Diagrammes de performance comparant les différentes méthodes.
- [ ] - Rédaction du rapport et dépôt sur le GitHub (date limite 6 février 2026)

## Instructions d'utilisation

Les données sont accessibles dans le répertoire [./data/](./data/). Pour lire et accéder aux données d'un fichier de données, exécuter la commande Julia `include("chemin/du/fichier")`.

Un parser de data est accessible dans le fichier [./src/parser.jl](./src/parser.jl), qui sert notamment à calculer la matrice des distances. Pour l'utiliser, il faut :
1. Inclure le fichier : `include("src/parser.jl")`
2. Exécuter la fonction Julia : `(n, L, W, K, B, w_v, W_v, lh, coordinates, distances) = parse()`

Après avoir suivi dans l'ordre les deux étapes suivantes, tous les paramètres sont normalement définis pour pouvoir appeler les différentes méthodes d'optimisation. Par exemple : `(optimum, y_opt) = solve_cutting_planes(n, L, W, K, B, w_v, W_v, lh, distances)`