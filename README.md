# Titanic_AI_vs_Human_Decision

![Titenic Dataset Analysis](https://github.com/almazid82/Titanic_AI_vs_Human_Decision/blob/main/Titenic%20.jpg)

![View Dataset from here ](https://github.com/almazid82/Titanic_AI_vs_Human_Decision/blob/main/Titanic.csv)

## A Data Story: "Last Voyage – A Tale of Fate, Class, and Choice"

**April 15, 1912.** The world’s grandest ship, Titanic, sank into the icy waters of the North Atlantic. Onboard were 2,200+ souls—some rich, some poor, some dreaming of a new world, and some simply returning home. But when the unthinkable happened, not everyone got an equal chance at survival. 

Who lived? Who didn’t? And why?

This dataset holds the last known information about those passengers. Today, we don’t just look at names and numbers. We bring in Artificial Intelligence to revisit that fateful night and ask: _Could survival have been predicted?_ And more importantly—_How would a human decide differently from a machine?_

Let’s dive into the Titanic dataset—not just as data scientists, but as human beings curious about fairness, fate, and decision-making.

---

## Model vs Human: Diverging Perspectives on Survival

After analyzing the Titanic dataset using a Random Forest Classifier, we found significant patterns in the following features:

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
| sex       | 0.272501    |
| fare      | 0.269387    |
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

![Feature Importance](Feature Importance.png)

## Model Update with Best Hyperparameters

The RandomForestClassifier model was updated with the best hyperparameters found from GridSearchCV:

- **max_depth:** 20
- **min_samples_split:** 10
- **n_estimators:** 300

### Updated Model Accuracy
The updated model achieved an accuracy of **83.80%**.

### Feature Importance Analysis for Updated Model

![feature importance](https://github.com/almazid82/Titanic_AI_vs_Human_Decision/blob/main/Feature%20Importance.png)

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



## Default Random Forest Model Performance

### Performance Metrics

- **Recall:** 0.76
- **AUC Score:** 0.89

### Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.83      | 0.86   | 0.85     | 105     |
| 1     | 0.79      | 0.76   | 0.77     | 74      |

**Accuracy:** 0.82

### Analysis
- **Precision:** For class 0, the precision is 0.83, indicating that 83% of the predictions for class 0 were correct. For class 1, the precision is 0.79.
- **Recall:** For class 0, the recall is 0.86, indicating that 86% of the actual class 0 instances were correctly identified. For class 1, the recall is 0.76.
- **F1-Score:** The F1-score combines precision and recall, and for class 0, it is 0.85, and for class 1, it is 0.77.
- **Support:** Support indicates the number of actual occurrences of the class in the dataset. There were 105 instances of class 0 and 74 instances of class 1.

**Summary:** The default Random Forest model shows a balanced performance across both classes, with an overall accuracy of 82%. The AUC score of 0.89 indicates a good ability 


## Model Performance Comparison

## Confusion Matrix (Random Forest Model)

![Confusion Matrix - Random Forest](https://github.com/almazid82/Titanic_AI_vs_Human_Decision/raw/main/confution%20matrix%20plot%20of%20a%20Random%20forest%20model.png)

এই confusion matrix টি Random Forest model-এর পারফরমেন্স বিশ্লেষণ করে। এটি থেকে দেখা যায় মডেলটি কোন ক্যাটাগরি গুলিকে সঠিকভাবে চিহ্নিত করতে পেরেছে এবং কোথায় misclassification হয়েছে। এই visualization টি accuracy, recall, ও precision metrics বোঝার জন্য কার্যকর।

### Default Random Forest Model

- **Recall:** 0.76
- **AUC Score:** 0.89
- **Accuracy:** 0.82

#### Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.83      | 0.86   | 0.85     | 105     |
| 1     | 0.79      | 0.76   | 0.77     | 74      |

### Hyperparameter Tuned Random Forest Model

- **Recall:** 0.77
- **AUC Score:** 0.90
- **Accuracy:** 0.84

#### Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.85      | 0.89   | 0.87     | 105     |
| 1     | 0.83      | 0.77   | 0.80     | 74      |

### Analysis
The hyperparameter tuned model shows improvement across all performance metrics compared to the default model. The increased recall and AUC score indicate a better ability to identify true positives and distinguish between classes. Overall accuracy has also improved.


## Confusion Matrix (After Hyperparameter Tuning)

![Confusion Matrix](https://github.com/almazid82/Titanic_AI_vs_Human_Decision/raw/main/confusion%20table%20for%20Hyperparameter%20Tuning.png)

এই চিত্রটি Hyperparameter tuning এর পর model এর performance বোঝাতে সাহায্য করে। True positives এবং false predictions স্পষ্টভাবে বোঝা যায় এই confusion matrix থেকে — যা model optimization-এর জন্য অত্যন্ত গুরুত্বপূর্ণ।



### Best Model Precision: 
**0.83**

### Best Model Recall: 
**0.77**

### Best Model AUC Score: 
**0.90**

### Classification Report for Best Model:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.85      | 0.89   | 0.87     | 105     |
| 1     | 0.83      | 0.77   | 0.80     | 74      |

**Overall Accuracy:** 0.84

**Macro Average:**
- Precision: 0.84
- Recall: 0.83
- F1-Score: 0.83

**Weighted Average:**
- Precision: 0.84
- Recall: 0.84
- F1-Score: 0.84



## SHAP Interaction Visualization

![SHAP Interaction](https://github.com/almazid82/Titanic_AI_vs_Human_Decision/raw/main/SHAP%20interaction%20value.png)

এই visualization টি SHAP interaction value বিশ্লেষণে সাহায্য করে, যেখানে আমরা feature গুলোর পারস্পরিক সম্পর্ক বুঝতে পারি এবং কোন feature decision-making এ সবচেয়ে বেশি প্রভাব রাখছে তা চিহ্নিত করতে পারি

### Human Perspective on Survival Decisions:
Humans, during the crisis, relied on **social rules and emotions**:

- **Women and Children First:** Most rescuers prioritized women and children—an ethical norm, not necessarily data-driven.
- **Visible Class Indicators:** First-class passengers were more visible and closer to lifeboats. They received priority not only due to proximity but also perceived importance.
- **Emotional Choices:** Families tried to stay together. Some refused rescue to stay with loved ones.

### AI Perspective on Survival Decisions:
AI, on the other hand, **relies solely on patterns in historical data**:

- **Sex** was the most important predictor (36.86% importance), showing that females had higher survival rates.
- **Fare** (21.90%) and **Pclass** (10.93%) highlighted socio-economic advantage.
- **Age** (18.15%) was significant; younger passengers were more likely to survive.
- **SibSp** and **Parch** showed minor but notable influences.

---

## Final Thoughts: Machine vs Man

| Perspective | Strengths | Weaknesses |
|-------------|-----------|------------|
| **Human**   | Emotionally aware, empathetic, moral judgment | Inconsistent, biased by class or relationships |
| **AI**      | Pattern recognition, consistent decision-making | Lacks empathy, may overfit past patterns, ignores context |

### Conclusion:
AI can analyze survival likelihood with high accuracy (~84%), but humans bring moral context and situational ethics. A rescue plan optimized by AI might be efficient—but would it be human? Meanwhile, human decision-making might be flawed, but it reflects values like sacrifice, love, and honor.

---

## Let’s Think Together:
- Would a fully AI-controlled evacuation have saved more lives?
- Or would it have neglected the very humanity that makes life worth saving?

This project is not just about models. It’s about merging logic and emotion to understand how we make decisions—then, and now.



## Future Scope: Applicability in Bangladesh’s Maritime Sector


Finally, Can this be applied for Bangladesh’s Maritime Sector?

Although the Titanic dataset is historical and foreign, the decision modeling techniques applied (e.g., logistic regression, decision trees) are universally applicable.

In the context of Bangladesh’s growing shipbuilding and passenger transport industry, AI-based decision systems can:

- Predict survival chances during maritime accidents.
- Help in safety protocol planning and risk assessment.
- Assist in designing smart evacuation systems using real-time data.

This study can serve as a prototype for building customized AI models using local maritime data in the future.

Collaboration with Bangladesh Inland Water Transport Authority (BIWTA) or ship companies can bring practical implementation.


## License

This content is shared for **educational and portfolio purposes only**.  
Licensed under the [MIT License](LICENSE) © 2025 Shamsul Al Mazid.

---

## Contact

**Shamsul Al Mazid**  
Aspiring Data Analyst  
Email: [shamsulalmazid@gmail.com]  
LinkedIn: [linkedin.com/in/shamsul-al-mazid](https://www.linkedin.com/in/shamsul-al-mazid-073a87286)

