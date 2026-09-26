from sklearn.datasets import load_diabetes
import pandas as pd


def main():
    # 1. Chargement du dataset
    diabetes = load_diabetes()

    # 2. Transformation en DataFrame pandas
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    # 3. Ajout de la variable cible
    data["target"] = diabetes.target

    # 4. Premiers affichages
    print("Aperçu des données :")
    print(data.head())

    print("\nDimensions du dataset :")
    print(data.shape)

    print("\nVariables disponibles :")
    print(data.columns.tolist())

    print("\nStatistiques descriptives :")
    print(data.describe())

    print("\nCorrélations avec la variable cible :")
    correlations = data.corr()["target"].sort_values(ascending=False)
    print(correlations)

if __name__ == "__main__":
    main()