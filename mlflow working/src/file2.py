import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


import dagshub
dagshub.init(repo_owner='VaibhavRai24', repo_name='Mlops-MLflow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/VaibhavRai24/Mlops-MLflow.mlflow")

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


max_depth = 10
n_estimators = 10

mlflow.set_experiment("Experiment 2")

with mlflow.start_run():
    mlflow.autolog()
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title('Confusion Matrix')
    
    plt.savefig("confusion_matrix.png")
    
    
    mlflow.log_artifact(__file__)
    
    
    mlflow.set_tags({"data": "wine", "model": "RandomForest"})
    print(accuracy)