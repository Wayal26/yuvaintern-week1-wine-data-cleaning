import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

wine=load_wine(as_frame=True)
df=wine.frame.copy()
X=df.drop(columns=['target']); y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)

models={
'Logistic Regression':Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=2000,random_state=42))]),
'Random Forest':RandomForestClassifier(n_estimators=300,random_state=42)
}
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for name,model in models.items():
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    print(name)
    print('Accuracy:',accuracy_score(y_test,pred))
    print('Precision:',precision_score(y_test,pred,average='weighted'))
    print('Recall:',recall_score(y_test,pred,average='weighted'))
    print('F1:',f1_score(y_test,pred,average='weighted'))
    print('CV accuracy:',cross_val_score(model,X_train,y_train,cv=cv,scoring='accuracy').mean())
    print(confusion_matrix(y_test,pred))
