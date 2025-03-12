# Titanic_AI_vs_Human_Decision
 "Analyzing AI vs Human Decision Making in Titanic Dataset

 # Titanic AI vs Human Decision

This project analyzes the Titanic dataset to predict the survival of passengers using a Random Forest Classifier.

## Features:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

## Instructions:
1. Clone the repository
2. Install necessary libraries (pandas, scikit-learn, matplotlib)
3. Run the Python script

## Model Performance:
The model achieved an accuracy of 81.56%.


## Feature Importance and Comparative Analysis with Random Forest

Below is the feature importance table for the Random Forest model:

| Feature   | Importance |
|-----------|-------------|
| Fare      | 0.272501    |
| Sex       | 0.269387    |
| Age       | 0.251845    |
| Pclass    | 0.087854    |
| SibSp     | 0.047876    |
| Parch     | 0.037125    |
| Embarked  | 0.033413    |


### Detailed Analysis

#### Fare
- The fare paid by passengers significantly impacts their survival. Higher fares often correspond to better living conditions and quicker access to lifeboats.

#### Sex
- Gender is a critical factor, with females generally having a higher chance of survival due to the "women and children first" policy during the evacuation.

#### Age
- Younger passengers are more likely to survive, possibly due to being prioritized during the rescue efforts.

#### Pclass
- Passenger class also plays a role in survival rates, with higher-class passengers having better access to lifeboats and safety measures.

#### Comparative Analysis
- Comparing these features with other studies and historical data, it is evident that social and economic factors played a significant role in the survival of Titanic passengers.

## Model Update with Best Hyperparameters

The RandomForestClassifier model was updated with the best hyperparameters found from GridSearchCV:

- **max_depth:** 20
- **min_samples_split:** 10
- **n_estimators:** 300

### Updated Model Accuracy
The updated model achieved an accuracy of **83.80%**.

### Feature Importance Analysis for Updated Model

| Feature  | Importance |
|----------|-------------|
| Sex      | 0.368605    |
| Fare     | 0.218975    |
| Age      | 0.181490    |
| Pclass   | 0.109297    |
| SibSp    | 0.047906    |
| Parch    | 0.038442    |
| Embarked | 0.035285    |
### Detailed Analysis

- **Sex:** Gender is the most important feature, contributing approximately 36.86% to the model's predictions. Historically, during the Titanic disaster, the "women and children first" policy was applied, leading to a higher survival rate for females.
- **Fare:** The ticket fare is the second most important feature, contributing approximately 21.90%. Higher fare-paying passengers generally had better living conditions and access to lifeboats.
- **Age:** Age is the third most important feature, contributing approximately 18.15%. Younger passengers had a higher likelihood of survival, possibly due to being more physically active and responsive.
- **Pclass:** Passenger class is also significant, contributing approximately 10.93%. Higher-class passengers had better access to safety measures.
- **SibSp:** Sibling/Spouse feature contributes around 4.79%. Passengers traveling with family members had a slightly higher chance of survival.
- **Parch:** Parent/Child feature contributes around 3.84%. Passengers traveling with parents or children had a slightly higher chance of survival.
- **Embarked:** The port of embarkation contributes around 3.53%. The departure port had a minor impact on survival chances.

This analysis provides insights into the importance of various features in predicting passenger survival on the Titanic.
