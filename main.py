""" Main for App Model Scikit-Learn"""

#Utils
from utils import Utils
#models
from models import Models

if __name__ == "__main__":

    utils = Utils()
    models = Models()

    data = utils.load_from_csv('./data/felicidad.csv')
    x, y = utils.features_target(data, ['score','rank', 'country'],['score'])

    models.grid_training(x,y)

    print(data)