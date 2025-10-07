import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("health.csv")

x = data.drop("HeartDisease", axis=1)

y = data["HeartDisease"]

xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(xtr, ytr)

ypr = rf.predict(xte)

acc = accuracy_score(yte, ypr)

print("Random Forest Accuracy:", acc)

print("\nClassification Report:")

print(classification_report(yte, ypr))

print("\nConfusion Matrix:")

print(confusion_matrix(yte, ypr))

from sklearn.tree import plot_tree

import matplotlib.pyplot as plt


# Visualize first 3 trees

for i in range(3):

    plt.figure(figsize=(5,4))
    plot_tree(rf.estimators_[i],
              feature_names=x.columns,
              class_names=["No Disease","Disease"],
              filled=True, rounded=True)
    plt.title(f"Decision Tree {i+1} from Random Forest")
    plt.show()
