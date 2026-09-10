import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)
df = wine.frame.copy()
df["target_name"] = df["target"].map(dict(enumerate(wine.target_names)))

print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df.describe().T)

df["target_name"].value_counts().sort_index().plot(kind="bar")
plt.title("Wine Class Distribution")
plt.xlabel("Wine Class")
plt.ylabel("Number of Samples")
plt.show()

for feature in ["alcohol", "malic_acid"]:
    plt.figure(figsize=(7,5))
    plt.hist(df[feature], bins=15)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

for feature in ["alcohol", "flavanoids"]:
    df.boxplot(column=feature, by="target_name")
    plt.suptitle("")
    plt.title(f"{feature.title()} by Wine Class")
    plt.xlabel("Wine Class")
    plt.ylabel(feature)
    plt.show()

numeric = df.drop(columns=["target", "target_name"])
corr = numeric.corr()
plt.figure(figsize=(10,8))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
