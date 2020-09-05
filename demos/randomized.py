""" Implementation Randomized Search CV"""
import pandas as pd

# sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":

    dataset = pd.read_csv('./data/felicidad.csv')

    print(dataset)

    x = dataset.drop(['country', 'rank', 'score'], axis=1)
    y = dataset[['score']]

    reg = RandomForestRegressor()

    parametros = {
        'n_estimators' : range(4,16),
        'criterion' : ['mse', 'mae'],
        'max_depth' : range(2,11)
    }

    rand_est = RandomizedSearchCV(reg, parametros , n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(x,y)

    print("-"*32)
    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    #print(rand_est.predict(x.loc[[0]]))

    y_hat = rand_est.predict(x.loc[[0]])
    print('Predict: {}'.format(y_hat[0]))
    print('Real:    {}'.format(y))