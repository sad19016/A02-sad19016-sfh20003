# Let's get started! We'll import the data first.

from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Now, let's build the boxplot!
plt.figure(figsize=(8, 6))
plt.boxplot(df['MedHouseVal'])
plt.title('Boxplot of MedHouseVal')
plt.ylabel('MedHouseVal')
plt.grid(True, alpha=0.3)

# Save it to the figures folder!
plt.savefig('../figures/boxplot.png', dpi=300, bbox_inches='tight')
plt.show()