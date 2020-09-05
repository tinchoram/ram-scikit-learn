import pandas as pd

#Sklearn
from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":

    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(10))

    x = dataset.drop('competitorname', axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(x)
    print("Total de centros: " , len(kmeans.cluster_centers_))
    print("-"*32)
    print(kmeans.predict(x))

    dataset['group'] = kmeans.predict(x)

    print(dataset)