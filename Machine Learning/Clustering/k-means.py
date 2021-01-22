import numpy as np
import time
from PIL import Image 


class Kmeans:
    def euclidean_dist(self, x, y):
        return np.linalg.norm(x - y)
    
    def calculate_distances(self, img, cluster_means):
        distances = np.zeros((img.shape[0], img.shape[1], cluster_means.shape[0]))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(cluster_means.shape[0]):
                    distances[i][j][k] = self.euclidean_dist(img[i][j], cluster_means[k])
                    
        return distances
    
    def assign_labels(self, img, distances, cluster_means):
        labels = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        cluster_pixels = [[] for i in range(cluster_means.shape[0])]
        error = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                label = np.argmin(distances[i][j])
                labels[i][j] = label
                cluster_pixels[label].append(img[i][j])
                error += self.euclidean_dist(img[i][j], cluster_means[label])
        
        error /= img.shape[0] * img.shape[1]
        return labels, cluster_pixels, error
    
    def update_cluster_means(self, cluster_means, cluster_pixels):
        for i in range(cluster_means.shape[0]):
            cluster_means[i] = np.mean(cluster_pixels[i], axis=0)
    
    def get_clustered_img(self, labels, cluster_means):
        img = np.zeros((labels.shape[0], labels.shape[1], cluster_means.shape[1]))
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                img[i][j] = cluster_means[labels[i][j]]
                
        return img
    
    def apply(self, img, k, error_limit):
        cluster_means = np.zeros((k, img.shape[2]))
        for i in range(k):
            x = np.random.randint(img.shape[0])
            y = np.random.randint(img.shape[1])
            cluster_means[i] = img[x][y]
        
        prev_error = np.inf
        while True:
            distances = self.calculate_distances(img, cluster_means)
            labels, cluster_pixels, error = self.assign_labels(img, distances, cluster_means)
            if prev_error - error < error_limit:
                break
            prev_error = error
            self.update_cluster_means(cluster_means, cluster_pixels)       
        
        print("Error:", error)
        print("Clustering vectors:", cluster_means)
        return self.get_clustered_img(labels, cluster_means)
        

img = np.asarray(Image.open('sample.jpg'))

kmeans = Kmeans()
for k in range(2, 11):
    print("K:", k)
    start_time = time.time()
    clustered_img_data = kmeans.apply(img, k, 0.01).astype(np.uint8)
    end_time = time.time()
    print("Execution time:", end_time-start_time)
