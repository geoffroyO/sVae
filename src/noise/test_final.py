import cv2
import matplotlib.pyplot as plt
import seaborn as sns;
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from pylab import savefig
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import skimage.morphology as morph
from scipy.ndimage.measurements import label
from tqdm import tqdm


if __name__=='__main__':
    res = []
    for noise in tqdm([20, 40, 60, 80, 100]):
        for k in tqdm(range(1, 11)):
            n = "{}_".format(noise) + "{}".format(k)
            reconstruction = np.load("./{}_reconstruction.npy".format(n))
            features = np.load("./{}_features.npy".format(n))

            error = np.abs(features - reconstruction)
            error = np.sum(error, axis=-1)
            error_list = []

            # K-means
            n, m = error.shape

            for i in range(n):
                for j in range(m):
                    error_list.append([error[i, j]])

            kmeans = KMeans(n_clusters=5, random_state=0).fit(error_list)
            counts = [0, 0, 0, 0, 0]
            # Count k-means class
            for e in kmeans.labels_:
                counts[e] += 1

            error_2 = np.zeros((n, m))
            count = 0
            for i in range(n):
                for j in range(m):
                    classe = kmeans.labels_[count]
                    if classe != np.argmax(counts):
                        if classe == 2:
                            error_2[i, j] = 1
                        elif classe == 3:
                            error_2[i, j] = 1

                    count += 1

            # Morphology
            closing = morph.binary_closing(error_2, morph.square(4))
            opening = morph.binary_opening(closing, morph.square(4))

            # CC
            labeled_array, num_features = label(opening)

            count = [0 for _ in range(num_features)]
            for i in range(n):
                for j in range(m):
                    lab = labeled_array[i, j]
                    if lab != 0:
                        count[lab - 1] += 1

            # take biggest cc
            error_final = np.zeros(error.shape)
            for i in range(n):
                for j in range(m):
                    if labeled_array[i, j] == np.argmax(count) + 1:
                        error_final[i, j] = 1

            closing_2 = morph.binary_closing(error_final, morph.square(15))
            dilatation = morph.binary_dilation(closing_2)
            res.append(dilatation)
    np.save("./res.npy", np.array(res))