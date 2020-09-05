import pandas as pd 
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/whr2017.csv')
    
    x = dataset[['gdp', 'family', 'lifexp', 'freedom' , 'corruption' , 'generosity', 'dystopia']]
    y = dataset[['score']]

    print(x.shape)
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    modelLinear = LinearRegression().fit(x_train, y_train)
    y_predict_linear = modelLinear.predict(x_test)

    modelLasso = Lasso(alpha=0.02).fit(x_train, y_train)
    y_predict_lasso = modelLasso.predict(x_test)

    modelRidge = Ridge(alpha=1).fit(x_train, y_train)
    y_predict_ridge = modelRidge.predict(x_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss: ", linear_loss)

    linear_lasso = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss: ", linear_lasso)

    linear_ridge = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge Loss: ", linear_ridge)

    print("="*32)
    print("Coef LASSO")
    print(modelLasso.coef_)

    print("="*32)
    print("Coef RIDGE")
    print(modelRidge.coef_)




