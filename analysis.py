from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

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


    print("\nRégression linéaire multiple : target ~ toutes les variables")

    # Toutes les variables explicatives
    X_multi = data.drop(columns=["target"])
    y_multi = data["target"]

    # Séparation train/test
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi,
        y_multi,
        test_size=0.2,
        random_state=42
    )

    # Création et entraînement du modèle
    multi_model = LinearRegression()
    multi_model.fit(X_train_multi, y_train_multi)

    # Prédictions
    y_pred_multi = multi_model.predict(X_test_multi)

    # Évaluation
    mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
    rmse_multi = root_mean_squared_error(y_test_multi, y_pred_multi)
    r2_multi = r2_score(y_test_multi, y_pred_multi)

    print(f"Intercept : {multi_model.intercept_:.3f}")
    print(f"MSE : {mse_multi:.3f}")
    print(f"RMSE : {rmse_multi:.3f}")
    print(f"R² : {r2_multi:.3f}")
    print("\nCoefficients du modèle multiple :")
    coefficients = pd.Series(
        multi_model.coef_,
        index=X_multi.columns
    ).sort_values(ascending=False)
    print(coefficients)
    print("\nVariables les plus importantes en valeur absolue :")
    coefficients_abs = coefficients.abs().sort_values(ascending=False)
    print(coefficients_abs)

    print("\nComparaison des modèles :")
    model_comparison = pd.DataFrame({
        "model": [
            "Simple linear regression",
            "Multiple linear regression"
        ],
        "mse": [
            mse,
            mse_multi
        ],
        "rmse": [
            rmse,
            rmse_multi
        ],
        "r2": [
            r2,
            r2_multi
        ]
    })

    print(model_comparison)

    print("\nSauvegarde du graphique de comparaison des modèles...")

    plt.figure()
    plt.bar(model_comparison["model"], model_comparison["rmse"])
    plt.ylabel("RMSE")
    plt.title("Comparaison des modèles par RMSE")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison_rmse.png")
    plt.close()

    print("Graphique sauvegardé dans outputs/model_comparison_rmse.png")
if __name__ == "__main__":
    main()