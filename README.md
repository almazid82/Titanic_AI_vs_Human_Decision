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


## Feature Importance and Comparative Analysis

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


