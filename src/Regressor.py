from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame


from sklearn.model_selection import train_test_split

# Features (X) and target (y)
X = df.drop("MedHouseVal", axis=1)   # target column
y = df["MedHouseVal"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Quick sanity check
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
