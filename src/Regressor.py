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


# Saarth here! Now, let's take the split data and run it in MLPRegressor!

# First, let's add the scaler:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now, let's run the model:
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    alpha=0.0005,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train_scaled, y_train)