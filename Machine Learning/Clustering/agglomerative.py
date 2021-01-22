import numpy as np
import time
from PIL import Image 


class Agglomerative:
    def euclidean_dist(self, x, y):
        return np.linalg.norm(x - y)
    
    def assign_labels(self, img, cluster_means):
        distances = np.zeros((img.shape[0], img.shape[1], cluster_means.shape[0]))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(cluster_means.shape[0]):
                    distances[i][j][k] = self.euclidean_dist(img[i][j], cluster_means[k])
        
        labels = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        error = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                label = np.argmin(distances[i][j])
                labels[i][j] = label
                error += self.euclidean_dist(img[i][j], cluster_means[label])
        
        error /= img.shape[0] * img.shape[1]
        return labels, error
    
    def get_clustered_img(self, labels, cluster_means):
        img = np.zeros((labels.shape[0], labels.shape[1], cluster_means.shape[1]))
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                img[i][j] = cluster_means[labels[i][j]]
                
        return img
    
    def apply(self, img, k):
        cluster_sums = []
        cluster_counts = []
        for i in range(300):
            x = np.random.randint(img.shape[0])
            y = np.random.randint(img.shape[1])
            cluster_sums.append(img[x][y].astype(np.int32))
            cluster_counts.append(1)
        
        while len(cluster_sums) > k:
            min_dist = np.inf
            min_i = min_j = 0
            for i in range(len(cluster_sums)):
                for j in range(i + 1, len(cluster_sums)):
                    mean_i = cluster_sums[i] // cluster_counts[i]
                    mean_j = cluster_sums[j] // cluster_counts[j]
                    dist = self.euclidean_dist(mean_i, mean_j)
                    if dist < min_dist:
                        min_dist = dist
                        min_i, min_j = i, j
            
            cluster_sums[min_i] = cluster_sums[min_i] + cluster_sums[min_j]
            cluster_counts[min_i] = cluster_counts[min_i] + cluster_counts[min_j]
            cluster_sums.pop(min_j)
            cluster_counts.pop(min_j)
        
        cluster_means = np.array([cluster_sums[i] / cluster_counts[i] for i in range(k)])
        labels, error = self.assign_labels(img, cluster_means)
        
        print("Error:", error)
        print("Clustering vectors:", cluster_means)
        return self.get_clustered_img(labels, cluster_means)
            

img = np.asarray(Image.open('sample.jpg'))

agglom = Agglomerative()
for k in range(2, 11):
    print("K:", k)
    start_time = time.time()
    clustered_img_data = agglom.apply(img, k).astype(np.uint8)
    end_time = time.time()
    print("Execution time:", end_time-start_time)
