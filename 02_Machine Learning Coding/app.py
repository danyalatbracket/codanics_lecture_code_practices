import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = sns.load_dataset("tips")

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["sex", "day"], drop_first=True)
# print(df.head(10))
# Define features and target
X = df[['total_bill', 'size', 'sex_Female', 'day_Fri', 'day_Sat', 'day_Sun']]
Y = df['tip']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Feature scaling (SVR is sensitive to feature scaling)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Define SVR model
model = SVR(kernel='rbf', C=10, gamma=0.1)

# Fit the model
model.fit(X_train, Y_train)

# Predict on test set
Y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Visualize predictions
import matplotlib.pyplot as plt

plt.scatter(Y_test, Y_pred, color='blue')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual Tips')
plt.ylabel('Predicted Tips')
plt.title('Actual vs Predicted Tips')
plt.show()
