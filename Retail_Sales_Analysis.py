import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Superstore.csv")

print(df.head())

# Dataset information
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Convert OrderDate
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["Month"] = df["OrderDate"].dt.month_name()

# Sales by Category
sales_category = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(6,4))
sales_category.plot(kind="bar")
plt.title("Sales by Category")
plt.ylabel("Sales")
plt.show()

# Profit by Region
profit_region = df.groupby("Region")["Profit"].sum()

plt.figure(figsize=(6,4))
profit_region.plot(kind="bar")
plt.title("Profit by Region")
plt.ylabel("Profit")
plt.show()

# Monthly Sales
monthly_sales = df.groupby("Month")["Sales"].sum()

plt.figure(figsize=(8,4))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.ylabel("Sales")
plt.show()

# Encode categorical columns
encoder = LabelEncoder()

for col in ["Region", "Category", "SubCategory", "Segment"]:
    df[col] = encoder.fit_transform(df[col])

# Features and target
X = df[["Quantity", "Discount", "Profit", "Region", "Category", "Segment"]]
y = df["Sales"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# Actual vs Predicted
result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": pred
})

print(result)