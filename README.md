# Diabetes Regression Analysis

Projet d'analyse de données santé en Python, basé sur le dataset `diabetes` de `scikit-learn`.

L'objectif est d'étudier les relations entre plusieurs variables médicales anonymisées et une variable cible liée à la progression du diabète. Le projet met en pratique des outils de data analysis, de corrélation et de régression linéaire.

## Objectifs

- Charger et explorer un dataset de santé.
- Calculer des statistiques descriptives.
- Étudier les corrélations avec la variable cible.
- Construire un premier estimateur par régression linéaire simple.
- Construire un estimateur par régression linéaire multiple.
- Comparer les modèles avec MSE, RMSE et R².
- Générer un graphique de comparaison des performances.

## Données

Le projet utilise le dataset `diabetes` disponible directement dans `scikit-learn`.

Le dataset contient 442 observations et 10 variables explicatives standardisées :

- `age`
- `sex`
- `bmi`
- `bp`
- `s1`
- `s2`
- `s3`
- `s4`
- `s5`
- `s6`

La variable cible est :

- `target`

Les variables explicatives sont standardisées, donc elles ne sont pas exprimées dans leurs unités médicales brutes.

## Méthodes utilisées

Deux modèles sont comparés :

1. Régression linéaire simple :

```text
target ~ bmi


2. Régréssion linéaire multiple : 
target ~ age + sex + bmi + bp + s1 + s2 + s3 + s4 + s5 + s6