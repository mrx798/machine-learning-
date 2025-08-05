import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns

# Load data
df = pd.read_csv('D:\mc\ex2\ex2.csv')
df.columns = df.columns.str.strip()  # Remove extra spaces

print(df.columns)  # Check actual column names

# Features and target
x = df[['Bed Rooms', 'Size', 'Age', 'ZipCode']]
y = df['Selling Price']

# Encode categorical column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['ZipCode'])], remainder='passthrough')
xen = ct.fit_transform(x)

# Split data
xtr, xte, ytr, yte = train_test_split(xen, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(xtr, ytr)

# Predict
ypr = model.predict(xte)
print(ypr)

# Coefficients & intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=yte, y=ypr, color='blue', s=100)
plt.plot([min(yte), max(yte)], [min(yte), max(yte)], 'r--')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# Correlation heatmap
sns.heatmap(df[['Bed Rooms', 'Size', 'Age']].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
