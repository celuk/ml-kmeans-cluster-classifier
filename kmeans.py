# BIL 470
# Odev 3
# Seyyid Hikmet Celik
# 181201047

import random

class KMeansClusterClassifier():
    # Nesne olusturuldugunda baslangic degerlerinin atandigi fonksiyon
    def __init__(self, n_cluster = 3):
        self.n_cluster = n_cluster
        self.centroid = 0
        # 100 iterasyon sonunda centroidler bulunsun.
        # Iterasyon sayisi degistirilebilir.
        self.iterations = 100
        self.centroids = []
    
    # Modeli egittigimiz fonksiyon, bir sey dönmüyor.
    def fit(self, X) -> None:
        # Centroidleri olusturuyorum.
        numoffeatures = len(X[0])

        # Agirliklari veriyorum.
        weights = [[0, 0]] * numoffeatures
        for i in range(numoffeatures):
            weights[i] = [min(list(map(list, zip(*X)))[i]), max(list(map(list, zip(*X)))[i])]

        for i in range(self.n_cluster): 
            centroid_list = []

            # centroidleri rastgele seedliyorum.
            random.seed(i)
            for j in range(numoffeatures):
                centroid_list.append(weights[j][0]  + random.random() * (weights[j][1] - weights[j][0])) 
            self.centroids.append(centroid_list)


        # Etiketleri guncelliyorum.
        self.centroid = 0
        
        features = []
        for i in range(numoffeatures):
            features.append(list(map(list, zip(*X)))[i])

        # Euclid distance
        labels = []
        for i in range(len(X)):
            mindist = 10000000000
            eachlabel = -1

            for j in range(self.n_cluster):
                clustdist = 0

                for k in range(numoffeatures):
                    clustdist += pow(features[k][i] - self.centroids[j][k], 2)

                clustdist = pow(clustdist, 0.5) # sqrt(clustdist)

                if mindist > clustdist:
                    eachlabel = j
                    mindist = clustdist
                    

            self.centroid += mindist
            labels.append(eachlabel)
        
        # Verilen iterasyon sayisi kadar centroid ve etiketleri guncelliyorum.
        for i in range(self.iterations):
            # Centroidleri guncelliyorum.
            centroid_list = []

            for j in range(self.n_cluster):
                centroid_list.append([0] * numoffeatures)

            c_count = [0] * self.n_cluster

            for index, element in enumerate(X):
                for j in range(self.n_cluster):
                    if labels[index] == j:
                        for k in range(numoffeatures):
                            centroid_list[j][k] += element[k]
                        c_count[j] += 1
            
            for j in range(self.n_cluster):
                if c_count[j] != 0:
                    centroid_list[j] = [k / c_count[j] for k in centroid_list[j]]

            self.centroids = centroid_list

            # Etiketleri guncelliyorum.
            self.centroid = 0
            
            features = []
            for i in range(numoffeatures):
                features.append(list(map(list, zip(*X)))[i])

            # Euclid distance
            labels = []
            for i in range(len(X)):
                mindist = 10000000000
                eachlabel = -1

                for j in range(self.n_cluster):
                    clustdist = 0

                    for k in range(numoffeatures):
                        clustdist += pow(features[k][i] - self.centroids[j][k], 2)

                    clustdist = pow(clustdist, 0.5) # sqrt(clustdist)

                    if mindist > clustdist:
                        eachlabel = j
                        mindist = clustdist

                self.centroid += mindist
                labels.append(eachlabel)
        return

    # Model egitildikten sonra egitilen modele gore obeklenmis etiket listesini donen fonksiyon
    def predict(self, X) -> list:
        # Etiketleri guncelliyorum.
        self.centroid = 0
        numoffeatures = len(X[0])

        features = []
        for i in range(numoffeatures):
            features.append(list(map(list, zip(*X)))[i])

        # Euclid distance
        labels   = []
        for i in range(len(X)):
            mindist = 10000000000
            eachlabel = -1

            for j in range(self.n_cluster):
                clustdist = 0
                for k in range(numoffeatures):
                    clustdist += pow(features[k][i] - self.centroids[j][k], 2)
                clustdist = pow(clustdist, 0.5) # sqrt(clustdist)

                if mindist > clustdist:
                    eachlabel = j
                    mindist = clustdist

            self.centroid += mindist
            labels.append(eachlabel)

        for i in range(len(labels)): 
            labels[i] = labels[i] + 1

            if labels[i] == self.n_cluster:
                labels[i] = 0

        return labels
