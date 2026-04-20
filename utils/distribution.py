import math
import numpy as np
from scipy.stats import multivariate_normal, multinomial

class Gaussian_Distribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def probability(self, x):
        exponent = -((x - self.mean) ** 2) / (2 * self.std ** 2)
        prob = math.exp(exponent) # if * 1 / (\sqrt{2pi} * std) when std -> 0 prob -> inf
        return prob
    
class N_Gaussian_Distribution:
    def __init__(self, dim, **kwargs):
        self.dim = dim
        self.gaussian_dist = [Gaussian_Distribution(0, 1) for _ in range(dim)]
        self.data_point = []
        self.N = 0
    
    def probability(self, x):
        if x.shape[0] != self.dim:
            raise "dimension not match."
        return math.prod([self.gaussian_dist[i].probability(x[i]) for i in range(self.dim)])
    
    def set_data_point(self, data_point, regular):
        self.data_point = data_point
        self.N = len(data_point)

        for i in range(self.dim):
            mean = np.mean(data_point[:, i])
            std = np.std(data_point[:, i])
            if std < regular:
                std = regular
            self.gaussian_dist[i].mean = mean
            self.gaussian_dist[i].std = std

class Poisson_Distribution:
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def probability(self, x):
        if x < 0 or not isinstance(x, int):
            return 0
        prob = (self.lambda_ ** x) * math.exp(-self.lambda_) / math.factorial(x)
        return prob
    
class N_Poisson_Distribution:
    def __init__(self, dim, **kwargs):
        self.dim = dim
        self.poisson_dist = [Poisson_Distribution(1) for _ in range(dim)]  # Default λ = 1.
        self.data_point = []
        self.N = 0
    
    def probability(self, x):
        if x.shape[0] != self.dim:
            raise ValueError("dimension not match.")
        return math.prod([self.poisson_dist[i].probability(int(x[i])) for i in range(self.dim)])
    
    def set_data_point(self, data_point, **kwargs):
        self.data_point = data_point
        self.N = len(data_point)

        for i in range(self.dim):
            lambda_ = np.mean(data_point[:, i])  # For Poisson, λ is the sample mean.
            self.poisson_dist[i].lambda_ = lambda_

class Multinomial_Distribution:
    def __init__(self, dim, n=10, p=None):
        self.n = n
        self.p = []
        for i in range(dim):
            self.p.append(1/dim)
    
    def probability(self, x):
        if self.p is None:
            raise ValueError("Probabilities have not been set.")
        if len(x) != len(self.p):
            raise ValueError("Dimension of x does not match dimension of p.")
        return multinomial.pmf(x, self.n, self.p)
    
    def set_data_point(self, data_point):
        # data_point is assumed to be a 2D array where rows are samples
        counts = np.sum(data_point, axis=0)
        total_counts = np.sum(counts)

class Gaussian_Mixture_Distribution:
    def __init__(self, dim, n_components):
        self.dim = dim
        self.n_components = n_components
        self.means = np.zeros((n_components, dim))
        self.covariances = np.array([np.eye(dim)] * n_components)  # Initialize as identity matrices.
        self.weights = np.ones(n_components) / n_components  # Initialize with uniform weights.
    
    def probability(self, x):
        probs = np.array([self.weights[k] * multivariate_normal.pdf(x, self.means[k], self.covariances[k])
                          for k in range(self.n_components)])
        return np.sum(probs)
    
    def set_data_point(self, data_point, **kwargs):
        self.data_point = data_point
        self.N = len(data_point)

        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.n_components).fit(data_point)
        self.means = kmeans.cluster_centers_
        
        for k in range(self.n_components):
            points_in_cluster = data_point[kmeans.labels_ == k]
            self.covariances[k] = np.cov(points_in_cluster, rowvar=False)
            self.weights[k] = len(points_in_cluster) / len(data_point)
