import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

#MultiLayer Perceptron

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

data = pd.read_csv("health.csv")

x = data.drop("HeartDisease", axis=1) # Features

y = data["HeartDisease"] # Target

xtr,xte,ytr,yte = train_test_split(x, y, test_size=0.3,

random_state=42, stratify=y)

scale = StandardScaler()

xtrscale = scale.fit_transform(xtr)

xtescale = scale.transform(xte)

svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)

svm.fit(xtrscale, ytr)

# Step 4: Hybrid (SVM decision scores -> ANN)

xtrscore = svm.decision_function(xtrscale).reshape(-1,1)

xtescore = svm.decision_function(xtescale).reshape(-1,1)

ann = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)

ann.fit(xtrscore, ytr)

# Step 5: Accuracy comparison

svmacc = accuracy_score(yte, svm.predict(xtescale))

hybridacc = accuracy_score(yte, ann.predict(xtescore))

print(f"SVM Accuracy: {svmacc*100:.2f}%")

print(f"Hybrid Accuracy: {hybridacc*100:.2f}%")



import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Get predictions

svmpred = svm.predict(xtescale)

hybridpred = ann.predict(xtescore)

# Step 2: Confusion matrices

svmcm = confusion_matrix(yte, svmpred)

hybridcm = confusion_matrix(yte, hybridpred)

# Step 3: Plot side by side

fig, axes = plt.subplots(1, 2, figsize=(10,4))

disp1 = ConfusionMatrixDisplay(confusion_matrix=svmcm, display_labels=[0,1])

disp1.plot(ax=axes[0], colorbar=False)

axes[0].set_title("SVM Confusion Matrix")

disp2 = ConfusionMatrixDisplay(confusion_matrix=hybridcm, display_labels=[0,1])

disp2.plot(ax=axes[1], colorbar=False)

axes[1].set_title("Hybrid (SVM+ANN) Confusion Matrix")

plt.tight_layout()

plt.show()
