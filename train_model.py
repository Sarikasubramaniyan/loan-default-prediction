import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("data.csv")

# Drop LoanID (not useful for prediction)
df.drop("LoanID", axis=1, inplace=True)

# Label encode categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
#x = df.drop("Loan_Default", axis=1)
#y = df["Loan_Default"]
X = df.drop("Default", axis=1)
y = df["Default"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open("loan_model.pkl", "wb"))

# Save the column names used during training
model_columns = X.columns.tolist()
pickle.dump(model_columns, open("model_columns.pkl", "wb"))

print("✅ Model trained and saved as loan_model.pkl")
