import pandas as pd
df = pd.read_csv("pima-indians-diabetes.csv")
df.head()
target = df.label
inputs = df.drop('label', axis ='columns')
inputs.columns[inputs.isna().any()]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)
len(X_train)
len(X_test)
len(inputs)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
model.score(X_test,y_test)
X_test[:10]
y_test[:10]
model.predict(X_test[:10])
model.predict_proba(X_test[:10])