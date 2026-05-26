import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/housing.csv"
)

print(df.head())
print(df.info())
print(df.describe())

plt.figure(figsize=(8,5))
sns.histplot(
    df["MedHouseVal"],
    kde=True
)
plt.title("House Value Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="MedInc",
    y="MedHouseVal",
    data=df
)
plt.title(
    "Income vs House Value"
)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.show()