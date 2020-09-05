import pandas as pd

#sklearn
from sklearn.cluster import MeanShift

if __name__ == "__main__":

    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(5))

    x = dataset.drop('competitorname', axis=1)

    meanshift = MeanShift().fit(x)
    print(max(meanshift.labels_))

    print("-"*32)
    print(meanshift.cluster_centers_)

    dataset['meanshift'] = meanshift.labels_
    
    print("-"*32)
    print(dataset)