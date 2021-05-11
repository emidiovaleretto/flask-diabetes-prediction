import pickle
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

df_raw = pd.read_csv('pima-indians-diabetes.csv', header=None)

df1 = df_raw.copy()

# rename columns
df1.rename(columns={
    0: 'pregnancy',
    1: 'glucose',
    2: 'diastolic_pressure',
    3: 'triceps_skinfold_thickness',
    4: 'Insulin',
    5: 'bmi',
    6: 'family_history_diabetes',
    7: 'age',
    8: 'diabetic'
}, inplace=True)

df2 = df1.copy()

mms = MinMaxScaler()
x = df2.iloc[:, :-1]
y = df2.iloc[:, -1]
normalised_features = mms.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(normalised_features, y,
                                                    test_size=0.30, random_state=42)

# KNN Algorithm
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

y_pred_knn = knn_classifier.predict(X_test)

# Decision Tree Algorithm
dt_classifier = DecisionTreeClassifier(random_state=1)
dt_classifier.fit(X_train, y_train)

y_pred_dt = dt_classifier.predict(X_test)

# MLP Network Algorithm
mlp_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(5, 10),
                               random_state=1)

mlp_classifier.fit(X_train, y_train)

y_pred_mlp = mlp_classifier.predict(X_test)

# Saving Model
with open('model_diabetes.pkl', 'wb') as file:
    pickle.dump(mlp_classifier, file)
    print('Model successfully saved.')
