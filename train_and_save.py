import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt


def build_and_train(csv_path: str, model_out: str = "models/model.joblib"):
    df = pd.read_csv(csv_path)
    target = "price"
    if target not in df.columns:
        raise ValueError("Expected 'price' column as target in CSV")

    X = df.drop(columns=[target])
    y = df[target]

    # Identify column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training model...")
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Test R^2: {score:.4f}")

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved model to {out_path}")

    plot = plt.figure()
    plt.scatter(y_test, model.predict(X_test), alpha=0.3)
    plt.xlabel("True Prices")
    plt.ylabel("Predicted Prices")
    plt.title("True vs Predicted House Prices")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.savefig(out_path.parent / "prediction_scatter.png")
    print(f"Saved prediction scatter plot to {out_path.parent / 'prediction_scatter.png'}")
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and save house price model")
    parser.add_argument("csv", nargs="?", default="data/Housing.csv", help="Path to training CSV")
    parser.add_argument("--out", default="models/model.joblib", help="Output path for saved model")
    args = parser.parse_args()
    build_and_train(args.csv, args.out)
