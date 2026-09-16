import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=[
        "Sepal Length",
        "Sepal Width",
        "Petal Length",
        "Petal Width"
    ]
)

summary = pd.DataFrame({
    "Sepal Length": [
        df["Sepal Length"].mean(),
        df["Sepal Length"].median(),
        df["Sepal Length"].mode().iloc[0],
        df["Sepal Length"].std(),
        df["Sepal Length"].quantile(0.25),
        df["Sepal Length"].quantile(0.50),
        df["Sepal Length"].quantile(0.75)
    ],
    "Sepal Width": [
        df["Sepal Width"].mean(),
        df["Sepal Width"].median(),
        df["Sepal Width"].mode().iloc[0],
        df["Sepal Width"].std(),
        df["Sepal Width"].quantile(0.25),
        df["Sepal Width"].quantile(0.50),
        df["Sepal Width"].quantile(0.75)
    ],
    "Petal Length": [
        df["Petal Length"].mean(),
        df["Petal Length"].median(),
        df["Petal Length"].mode().iloc[0],
        df["Petal Length"].std(),
        df["Petal Length"].quantile(0.25),
        df["Petal Length"].quantile(0.50),
        df["Petal Length"].quantile(0.75)
    ],
    "Petal Width": [
        df["Petal Width"].mean(),
        df["Petal Width"].median(),
        df["Petal Width"].mode().iloc[0],
        df["Petal Width"].std(),
        df["Petal Width"].quantile(0.25),
        df["Petal Width"].quantile(0.50),
        df["Petal Width"].quantile(0.75)
    ]
}, index=[
    "Mean",
    "Median",
    "Mode",
    "Standard Deviation",
    "25th Percentile",
    "50th Percentile",
    "75th Percentile"
])

print(summary.round(2).to_string())