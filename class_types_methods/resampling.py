import numpy as np
from sklearn.neighbors import NearestNeighbors

class BaseResampling():
    def __init__(self, X, y, similarities_matrix=None, k=5):
        self.X = X
        self.y = y
        self.k = k
        self.min_classes, self.maj_classes = self.get_min_maj_classes(y)
        self.similarities_matrix = similarities_matrix if similarities_matrix is not None else self.get_similarities_matrix(y)
        self.nn_model = NearestNeighbors(n_neighbors=self.k+1).fit(X)
        
    def get_min_maj_classes(self, y):
        labels, counts = np.unique(y, return_counts=True)
        
        min_classes = []
        maj_classes = []
        
        for label, count in zip(labels, counts):
            if count < counts.sum() / counts.shape[0]:
                min_classes.append(label)
            else:
                maj_classes.append(label)
                
        return min_classes, maj_classes
    
    def get_similarities_matrix(self, y):
        labels, counts = np.unique(y, return_counts=True)
        
        return np.array([[min(i, j) / max(i, j) for j in counts] for i in counts])
    
class SimilarityOversampling(BaseResampling):
    def __init__(self, X, y, similarities_matrix=None, k=5):
        super().__init__(X, y, similarities_matrix, k)
        
    def run(self):        
        labels, counts = np.unique(self.y, return_counts=True)      
        diff = counts[np.isin(labels, self.maj_classes)] - counts[np.isin(labels, self.min_classes)]
        
        new_X = [self.X[np.isin(self.y, self.maj_classes)]]
        new_y = [self.y[np.isin(self.y, self.maj_classes)]]
        
        while diff > 0:
            D = []
            D_y = []
            D_sl = []
            
            for j in self.min_classes:
                D_j = self.X[np.argwhere(self.y == j).ravel()]
                neighbours_distances, neighbours_indices = self.nn_model.kneighbors(D_j)
                neighbours_distances = neighbours_distances[:, 1:]
                neighbours_indices = neighbours_indices[:, 1:]
                similarities = self.similarities_matrix[j]
                neighbours_classes = np.take(self.y, neighbours_indices)

                safe_levels = np.take(similarities, neighbours_classes).sum(axis=1) / (self.k + 1)
                
                D.append(D_j)
                D_y.append([j] * (D_j.shape[0]))
                D_sl.append(safe_levels)
                    
            D = np.vstack(D)
            D_y = np.hstack(D_y)
            D_sl = np.hstack(D_sl)
            
            sorted_indices = np.argsort(D_sl)
            new_X.append(D[sorted_indices > D.shape[0] - diff])
            new_y.append(D_y[sorted_indices > D.shape[0] - diff])
            
            diff -= D.shape[0]
                        
        self.new_X = np.vstack(new_X)
        self.new_y = np.hstack(new_y).reshape((-1, 1))
        
        return self.new_X, self.new_y
    
class SimilarityUndersampling(BaseResampling):
    def __init__(self, X, y, similarities_matrix=None, k=5):
        super().__init__(X, y, similarities_matrix, k)
        
    def run(self):        
        labels, counts = np.unique(self.y, return_counts=True)      
        diff = counts[np.isin(labels, self.maj_classes)] - counts[np.isin(labels, self.min_classes)]
        
        D = []
        D_y = []
        D_sl = []

        for i in self.maj_classes:
            D_i = self.X[np.argwhere(self.y == i).ravel()]
            neighbours_distances, neighbours_indices = self.nn_model.kneighbors(D_i)
            neighbours_distances = neighbours_distances[:, 1:]
            neighbours_indices = neighbours_indices[:, 1:]
            similarities = self.similarities_matrix[i]
            neighbours_classes = np.take(self.y, neighbours_indices)

            safe_levels = np.take(similarities, neighbours_classes).sum(axis=1) / (self.k + 1)

            D.append(D_i)
            D_y.append([i] * (D_i.shape[0]))
            D_sl.append(safe_levels)
                    
        D = np.vstack(D)
        D_y = np.hstack(D_y)
        D_sl = np.hstack(D_sl)

        sorted_indices = np.argsort(D_sl)
                        
        self.new_X = np.vstack([self.X[np.isin(self.y, self.min_classes)], D[sorted_indices > diff]])
        self.new_y = np.hstack([self.y[np.isin(self.y, self.min_classes)], D_y[sorted_indices > diff]])
        
        return self.new_X, self.new_y
    
class SOUP(BaseResampling):
    def __init__(self, X, y, similarities_matrix=None, k=5):
        super().__init__(X, y, similarities_matrix, k)
    
    def run(self):
        D = []
        D_y = []
        
        labels, counts = np.unique(self.y, return_counts=True)        
        m = np.mean([np.min(counts[np.isin(labels, self.maj_classes)]), np.max(counts[np.isin(labels, self.min_classes)])])
        
        for i in self.maj_classes:
            D_i = self.X[np.argwhere(self.y == i).ravel()]
            neighbours_distances, neighbours_indices = self.nn_model.kneighbors(D_i)
            neighbours_distances = neighbours_distances[:, 1:]
            neighbours_indices = neighbours_indices[:, 1:]
            similarities = self.similarities_matrix[i]
            neighbours_classes = np.take(self.y, neighbours_indices)

            safe_levels = np.take(similarities, neighbours_classes).sum(axis=1) / (self.k + 1)
            sorted_indices = np.argsort(safe_levels)
            to_remove = D_i.shape[0] - m
            
            D_not_removed = D_i[sorted_indices > to_remove]
            D.append(D_not_removed)
            D_y.append([i] * D_not_removed.shape[0])
            
        for j in self.min_classes:
            D_j = self.X[np.argwhere(self.y == j).ravel()]
                        
            while D_j.shape[0] < m - 1:
                neighbours_distances, neighbours_indices = self.nn_model.kneighbors(D_j)
                neighbours_distances = neighbours_distances[:, 1:]
                neighbours_indices = neighbours_indices[:, 1:]
                similarities = self.similarities_matrix[j]
                neighbours_classes = np.take(self.y, neighbours_indices)

                safe_levels = np.take(similarities, neighbours_classes).sum(axis=1) / (self.k + 1)
                sorted_indices = np.argsort(safe_levels)
                to_duplicate = m - D_j.shape[0]

                D_j = np.vstack([D_j, D_j[sorted_indices > (D_j.shape[0] - to_duplicate)]])
            
            D.append(D_j)
            D_y.append([j] * D_j.shape[0])
            
        self.new_X = np.vstack(D)
        self.new_y = np.hstack(D_y).reshape((-1, 1))
        
        return self.new_X, self.new_y