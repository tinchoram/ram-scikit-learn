import pandas as pd

#sklearn
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":

    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())

    x = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.35, random_state=42)

    knn_class = KNeighborsClassifier().fit(x_train, y_train)
    knn_pred = knn_class.predict(x_test)

    print('-'*32)
    print('Accuracy KNeighbors:', accuracy_score(knn_pred, y_test))
    print('-'*32)

    #bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(x_train, y_train)
    #bag_pred = bag_class.predict(x_test)

    #print('-'*32)
    #print('Accuracy Bagging with KNeighbors:', accuracy_score(bag_pred, y_test))
    #print('-'*32)

    classifier = {
        'KNeighbors': KNeighborsClassifier(),
        'LinearSCV': LinearSVC(),
        'SVC': SVC(),
        'SGDC': SGDClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }

    for name, estimator in classifier.items():
        bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=5).fit(x_train, y_train)
        bag_pred = bag_class.predict(x_test)

        print('-'*32)
        print('Accuracy Bagging with {}:'.format(name), accuracy_score(bag_pred, y_test))
        print('-'*32)
