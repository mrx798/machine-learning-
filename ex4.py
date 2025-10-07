import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('D:\mc\ex4\ex4.csv')
df = pd.DataFrame(data)

# Features and target
x = df[['study hours', 'attendance']]
y = df['result']

# Train model
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(x, y)

# Plot decision tree
plt.figure(figsize=(8, 6))
plot_tree(clf, feature_names=['study hours', 'attendance'], class_names=['0', '1'], filled=True)
plt.show()

# Predict for new data
new = [[5, 85]]
pred = clf.predict(new)
print('Prediction for new student:', '1' if pred[0] == 1 else '0')
