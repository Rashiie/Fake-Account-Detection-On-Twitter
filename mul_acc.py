import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

### load and preprocess data
##df = pd.read_csv('bb.csv')
####df = df[['age', 'bp', 'al', 'su', 'sc', 'sod', 'hemo', 'appet', 'class']]
##df.dropna(inplace=True)
##X = df.drop('Fake Or Not Category', axis=1)
##y = df['Fake Or Not Category']
####X = pd.get_dummies(X, columns=['appet'], drop_first=True)
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
####print(X_train.shape)
### train GBM model
##gbm = GradientBoostingClassifier(learning_rate=0.1, n_estimators=2, random_state=10)
##gbm.fit(X_train, y_train)
##gbm_pred = gbm.predict(X_test)
##gbm_score = accuracy_score(y_test, gbm_pred)
##
### train SVM model
##svm = SVC(kernel='linear', probability=True)
##svm.fit(X_train, y_train)
##svm_pred = svm.predict(X_test)
##svm_score = accuracy_score(y_test, svm_pred)
##
### train Random Forest model
##rf = RandomForestClassifier(max_depth=1, min_samples_split=42, random_state=100)
##rf.fit(X_train, y_train)
##rf_pred = rf.predict(X_test)
##rf_score = accuracy_score(y_test, rf_pred)
##
### train Logistic Regression model
##lr = LogisticRegression()
##lr.fit(X_train, y_train)
##lr_pred = lr.predict(X_test)
##lr_score = accuracy_score(y_test, lr_pred)
##
### create voting classifier with GBM, SVM, Random Forest, and Logistic Regression models
##hybrid_clf = VotingClassifier(
##    estimators=[('gbm', gbm), ('svm', svm), ('rf', rf), ('lr', lr)],
##    voting='hard'
##)
##
### fit voting classifier on training data
##hybrid_clf.fit(X_train, y_train)
##hybrid_pred = voting_clf.predict(X_test)
##hybrid_score = accuracy_score(y_test, voting_pred)
##
### print accuracy scores
##print(f"Gradient Boosting Classifier Accuracy: {gbm_score:.4f}")
##print(f"SVM Classifier Accuracy: {svm_score:.4f}")
##print(f"Random Forest Classifier Accuracy: {rf_score:.4f}")
##print(f"Logistic Regression Classifier Accuracy: {lr_score:.4f}")
##print(f"Voting Classifier Accuracy: {hybrid_score:.4f}")
##
### save the trained model to disk
##with open('hybrid.pkl', 'wb') as f:
##    pickle.dump(hybrid_clf, f)

# load the saved model from disk
with open('hybrid.pkl', 'rb') as f:
    hybrid_clf = pickle.load(f)

# test on one new data point
new_data = [[1,166,994,945,165,46,85]]
##new_df = pd.DataFrame(new_data, columns=X.columns)
##new_df = pd.get_dummies(new_df, columns=['appet'], drop_first=True)
new_pred = hybrid_clf.predict(new_data)
print(f"Prediction for new data: {new_pred[0]}")
