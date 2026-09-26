from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

    print("\nRégression linéaire simple : target ~ bmi")

    # Variable explicative X et variable cible y
    X = data[["bmi"]]
    y = data["target"]

    # Séparation entre données d'entraînement et données de test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Création et entraînement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Évaluation du modèle
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Intercept : {model.intercept_:.3f}")
    print(f"Coefficient bmi : {model.coef_[0]:.3f}")
    print(f"MSE : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R² : {r2:.3f}")

if __name__ == "__main__":
    main()