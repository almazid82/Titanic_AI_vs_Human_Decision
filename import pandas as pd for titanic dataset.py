import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Titanic Dataset Load
df = pd.read_csv("C:/Users/shams/Downloads/Titanic.csv")   # Ensure the dataset is in the correct directory

# প্রথম কয়েকটা ডেটা দেখানো
print(df.head())

# মিসিং ভ্যালু চেক করা
print(df.isnull().sum())

# Age-এর Missing Value Median দিয়ে পূরণ
df['Age'].fillna(df['Age'].median(), inplace=True)

# Embarked-এর Missing Value Mode দিয়ে পূরণ
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# এখানে আমরা কেবিন কলাম বাদ দিয়েছি কারন-যখন Cabin-এর সাথে Survival-এর relation কম থাকে,যখন Missing Value 70%+ এর বেশি হয় এবং যখন Feature বেশি দরকার নেই।
df['Has_cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
df.drop(columns=['Cabin'], inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# প্রয়োজনীয় Feature নির্বাচন
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df.dropna()  # Null Value Remove

# Categorical Value Encoding
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = df[features]
y = df["Survived"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction & Accuracy Check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

import pandas as pd
import matplotlib.pyplot as plt

# Feature Importance বের করা
importance = model.feature_importances_

# Feature Names এর সাথে Importance যুক্ত করা
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})

# Importances অনুযায়ী সাজানো
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print করা
print(feature_importance_df)

# Visualization
plt.figure(figsize=(10, 5))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.title('Feature Importance Analysis')
plt.gca().invert_yaxis()  # বড় Importance উপরে দেখানোর জন্য
plt.show()