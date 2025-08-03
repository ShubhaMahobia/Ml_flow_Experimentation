import dagshub
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dagshub.init(repo_owner='ShubhaMahobia', repo_name='Ml_flow_Experimentation', mlflow=True)



wine = load_wine()
x = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=42)

max_depth = 10
n_estimators = 10
mlflow.set_tracking_uri('https://dagshub.com/ShubhaMahobia/Ml_flow_Experimentation.mlflow')

mlflow.set_experiment('NEW EXPERIMENT')

mlflow.autolog(log_models=False)


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth= max_depth, n_estimators = n_estimators, random_state = 42)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)

    accuracyscore = accuracy_score(y_test, y_pred)


    mlflow.set_tags({'Project' : 'New'})



    #Confusion Metrics
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel("actual")
    plt.xlabel('Pred')
    plt.title('Confusion Metrics')

    plt.savefig("Confusion_Metrics.png")
    mlflow.log_artifact(__file__)