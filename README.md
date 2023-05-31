# credit-card-fraud-detection
This GitHub repository contains code for credit card fraud prevention using machine learning methods and deep learning techniques. The goal of this project is to develop models that can accurately detect fraudulent credit card transactions, helping to prevent financial losses for individuals and organizations.

Dataset
The dataset used for this project was obtained from Kaggle. The link to the dataset can be found in the notebook files.The dataset contains various features related to credit card transactions.

Machine Learning Methods
The CreditCardFraudDetectionusingMachineLearning.ipynb notebook focuses on applying machine learning algorithms to the credit card fraud detection problem. It starts with exploratory data analysis (EDA) to gain insights into the dataset and preprocesses the data for modeling purposes.
After preprocessing, the notebook trains and evaluates several classifier models, including Decision Tree Classifier, Random Forest Classifier, Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbors Classifier, and Gradient Boosting Classifier. The performance of each model is compared, and the Decision Tree Classifier is identified as the best performing one.
The hyperparameters of the Decision Tree Classifier are then tuned using appropriate techniques, and the model is tested with unseen data. The evaluation results demonstrate the model's high accuracy and reasonable performance in distinguishing fraudulent and non-fraudulent transactions.

Deep Learning Methods
The CreditCardFraudDetectionusingDeepLearning.ipynb notebook focuses on utilizing deep learning techniques, specifically feedforward and convolutional neural networks (CNNs), for credit card fraud prediction. Similar to the machine learning notebook, it begins with EDA and preprocessing steps on the dataset.
The notebook then builds and trains a feedforward neural network and a CNN model. The performance of both models is evaluated using test data, showcasing their impressive results. The CNN model achieves a higher accuracy and lower loss, indicating its superior performance in classifying fraudulent and non-fraudulent transactions.

Conclusion
In conclusion, this project demonstrates the use of machine learning and deep learning techniques for credit card fraud prevention. The Decision Tree Classifier showed impressive accuracy in the machine learning approach, while the CNN model outperformed in the deep learning approach.
The Decision Tree Classifier achieved an accuracy of 0.996188, indicating its ability to accurately classify most credit card transactions as fraudulent or non-fraudulent. The CNN model achieved a test loss of 0.024 and a test accuracy of 99.48%, demonstrating exceptional performance in accurately identifying fraudulent transactions. Overall, this project provides valuable insights into credit card fraud prevention using machine learning and deep learning methods, showcasing the potential of these techniques to aid in fraud detection and protect individuals and organizations from financial losses.
