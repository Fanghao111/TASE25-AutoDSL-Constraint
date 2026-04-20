import numpy as np
from scipy import stats
import scipy
from scipy.stats import multivariate_normal
import scipy.linalg
import scipy.special
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

class GaussianMixtureModel:
    def __init__(self, D=2, num_components=3, s0=5.0, ss=0.65):  
        self.num_components = num_components  # Number of Gaussian components in the GMM.
        self.D = D  # Data dimensionality.
        self.components = []

        # Initialize each Gaussian component.
        for _ in range(num_components):
            component = {
                'weight': 1.0 / num_components,  # Start with uniform weights.
                'mean': np.zeros(shape=(D, 1)),  # Start from the zero mean.
                'covariance': np.eye(D),  # Start from the identity covariance.
                'cholesky': np.eye(D),  # Initial Cholesky factor.
                's0': s0,
                'ss': ss,
                'rr': 1.0 / (s0**2/ss**2),
                'nn': 0,
                'vv': 5
            }
            component['cholesky'] = scipy.linalg.cholesky(component['covariance'] * component['vv'] + component['rr'] * component['mean'] @ component['mean'].T)
            self.components.append(component)

    def add_point(self, x):
        for component in self.components:
            component['nn'] += 1
            component['rr'] += 1
            component['vv'] += 1
            component['cholesky'] = self.__cholupdate(component['cholesky'], x, '+')
            component['mean'] += x
            component['weight'] = component['nn'] / (self.num_components + component['vv'])

    def del_point(self, x):
        for component in self.components:
            component['nn'] -= 1
            component['rr'] -= 1
            component['vv'] -= 1
            component['cholesky'] = self.__cholupdate(component['cholesky'], x, '-')
            component['mean'] -= x
            component['weight'] = component['nn'] / (self.num_components + component['vv'])

    def logpredictive(self, x):
        log_probs = []
        for component in self.components:
            dd = self.D
            nn = component['nn']
            rr = component['rr']
            vv = component['vv']
            cc = component['cholesky']
            xx = component['mean']
            lp = self.__z(dd, nn + 1, rr + 1, vv + 1, self.__cholupdate(cc, x), xx + x) \
                - self.__z(dd, nn, rr, vv, cc, xx)
            log_probs.append(lp)
        # Compute the weighted sum of component log-probabilities.
        total_log_prob = np.log(sum(component['weight'] * np.exp(lp) for component, lp in zip(self.components, log_probs)))
        return total_log_prob

    def __cholupdate(self, R_current, x_update, sign='+'):
        R = R_current.copy()
        x = x_update.copy()
        n = len(x)

        for k in range(n):
            if sign == '+':
                r = np.sqrt(R[k, k]**2 + x[k, 0]**2)
            else:
                r = np.sqrt(R[k, k]**2 - x[k, 0]**2)
            c = r / R[k, k]
            s = x[k, 0] / R[k, k]

            R[k,k] = r

            if k !=n-1:
                if sign == '+':
                    R[(k+1):n, k] = (R[(k+1):n, k] + s * x[(k+1):n, 0]) / c
                else:
                    R[(k+1):n, k] = (R[(k+1):n, k] - s * x[(k+1):n, 0]) / c
                x[(k+1):n, 0] = c * x[(k+1):n, 0] - s * R[(k+1):n, k]
        return R

    def __z(self, dd, nn, rr, vv, cc, xx):
        zz = - nn*dd/2*np.log(np.pi) - dd/2*np.log(rr) - \
            vv*np.sum(np.log(np.diag(self.__cholupdate(cc, xx / np.sqrt(rr),'-')))) \
            + np.sum(scipy.special.loggamma((vv-(np.arange(0,dd, 1)))/2))
        return zz

class MultivariateNormal:
    def __init__(self, D=2, s0 = 5.0, ss = 0.65):       
        self.dd = D # Dimension
        self.vv = D # Prior effective sample size
        self.vc = np.eye(D) # Prior covariance
        self.uu = np.zeros(shape=(D, 1)) # Prior mean
        self.nn = 0 # Number of data points
        self.ws = self.vc * self.vv # Weighted sum of data points
        self.rr = 1.0 / (s0**2/ss**2) # Prior precision
        self.cc = scipy.linalg.cholesky(self.ws + self.rr * self.uu @ self.uu.T) # Cholesky decomposition of the covariance matrix
        self.xx = np.zeros(shape=(D,1)) # Sum of data points

    def add_point(self, x):
        # Add a data point
        self.nn += 1
        self.rr += 1
        self.vv += 1
        self.cc = self.__cholupdate(self.cc, x, '+')
        self.xx += x
    
    def del_point(self, x):
        # Remove a data point
        self.nn -= 1
        self.rr -= 1
        self.vv -= 1
        self.cc = self.__cholupdate(self.cc, x, '-')
        self.xx -= x

    def logpredictive(self, x):
        dd = self.dd
        nn = self.nn
        rr = self.rr
        vv = self.vv
        cc = self.cc
        xx = self.xx
        lp = self.__z(dd,nn+1,rr+1,vv+1,self.__cholupdate(cc,x),xx+x) \
             - self.__z(dd, nn, rr, vv ,cc ,xx)
        return lp

    def __cholupdate(self, R_current, x_update, sign='+'):
        # Cholesky decomposition update
        R = R_current.copy()
        x = x_update.copy()
        n = len(x)

        for k in range(n):
            if sign == '+':
                r = np.sqrt(R[k, k]**2 + x[k, 0]**2)
            else:
                r = np.sqrt(abs(R[k, k]**2 - x[k, 0]**2))
            c = r / R[k, k]
            s = x[k, 0] / R[k, k]

            R[k,k] = r

            if k !=n-1:
                if sign == '+':
                    R[(k+1):n, k] = (R[(k+1):n, k] + s * x[(k+1):n, 0]) / c
                else:
                    R[(k+1):n, k] = (R[(k+1):n, k] - s * x[(k+1):n, 0]) / c
                x[(k+1):n, 0] = c * x[(k+1):n, 0] - s * R[(k+1):n, k]
        return R

    def __z(self, dd, nn, rr, vv, cc, xx):
        # Log predictive distribution
        values = (vv - np.arange(0, dd, 1)) / 2
        valid_values = values[values > 0]

        zz = - nn*dd/2*np.log(np.pi) - dd/2*np.log(rr) - \
             vv*np.sum(np.log(np.diag(self.__cholupdate(cc, xx / np.sqrt(rr),'-')))) \
             + np.sum(scipy.special.loggamma(values))
        
        return zz

class Multinomial:
    def __init__(self, D=2, alpha=1.0):
        self.D = D  # Number of event types.
        self.alpha = np.full(D, alpha)  # Dirichlet prior parameters for smoothing.
        self.counts = np.zeros(D)  # Event counts.
        self.total_count = 0  # Total count.

    def add_point(self, x):
        """Add one data point by increasing the event counts."""
        self.counts += x
        self.total_count += np.sum(x)

    def del_point(self, x):
        """Remove one data point by decreasing the event counts."""
        self.counts -= x
        self.total_count -= np.sum(x)

    def logpredictive(self, x):
        """Compute the log predictive probability for a new data point."""
        log_pred = scipy.special.loggamma(self.alpha + self.counts + x).sum() \
                 - scipy.special.loggamma(self.alpha + self.counts).sum() \
                 + scipy.special.loggamma(self.alpha.sum() + self.total_count) \
                 - scipy.special.loggamma(self.alpha.sum() + self.total_count + np.sum(x))
        return log_pred

class PymcMultivariateNormal:
    def __init__(self, D, data):
        self.D = D
        self.data = data

    def fitting(self):
        with pm.Model() as model:
            # Prior over the mean of an N-dimensional Gaussian.
            mu = pm.Normal('mu', mu=np.zeros(self.D), sigma=np.ones(self.D)*10, shape=self.D)
            
            # Define the N-dimensional covariance matrix with an LKJ Cholesky prior.
            chol, corr, stds = pm.LKJCholeskyCov('chol', n=self.D, eta=2, sd_dist=pm.Exponential.dist(1.0), compute_corr=True)
            cov = pm.Deterministic('cov', chol @ chol.T)
            
            # Likelihood under the observed multivariate normal samples.
            likelihood = pm.MvNormal('likelihood', mu=mu, chol=chol, observed=self.data)
            
            # Draw posterior samples.
            trace = pm.sample(2000, return_inferencedata=True)

        self.trace = trace

    def predict(self, x):
        # Estimate the probability of a new point.
        with pm.Model() as model:
            ppc = pm.sample_posterior_predictive(self.trace, var_names=["mu", "cov"], samples=1000)

        # Extract posterior parameter estimates.
        mu_samples = ppc.posterior_predictive['mu'].mean(axis=0)
        cov_samples = ppc.posterior_predictive['cov'].mean(axis=0)

        # Compute the probability density of the new point.
        from scipy.stats import multivariate_normal
        probability_density = multivariate_normal.pdf(x, mean=mu_samples, cov=cov_samples)

        print("Probability density of the new point:", probability_density)

    def del_point(self, x):
        index = np.where(np.all(self.data == x, axis=1))
        self.data = np.delete(self.data, index, axis=0)
        self.fitting()
    
    def add_point(self, x):
        self.data = np.vstack([self.data, x])
        self.fitting()

def DPMM(X, Model=MultivariateNormal, K=2, z_init=None, alpha=1.0, max_iters=200):
    # transfer 2-dimension list X to numpy array
    for i in range(len(X)):
        X[i] = np.array(X[i])
    X = np.array(X)
    N = len(X)
    D = X.shape[1]

    if z_init is None:
        z = np.random.randint(0, K, size=N)
    else:
        z = z_init
        K = len(np.unique(z))
    n_points_cluster = [0] * K

    clusters = [Model(D=D) for _ in range(K)]
    dummy_dist = Model(D=D)


    for i in range(N):
        c = z[i]
        n_points_cluster[c] += 1
        clus = clusters[c]
        clus.add_point(X[i].reshape(D, 1))

    for _ in trange(max_iters, desc="Processing", leave=False):
        for i in range(N):
            z_i = z[i]
            x_i = X[i].reshape(D, 1)

            current_clus = clusters[z_i]

            current_clus.del_point(x_i)
            n_points_cluster[z_i] -= 1

            if n_points_cluster[z_i] == 0:
                del n_points_cluster[z_i]
                del clusters[z_i]
                z[z > z_i] -= 1
            prob = np.log(np.array(n_points_cluster + [alpha]))
            for j in range(len(clusters)):
                prob[j] = prob[j] + clusters[j].logpredictive(x_i)
            prob[-1] = prob[-1] + dummy_dist.logpredictive(x_i)
            prob = np.exp(prob - np.max(prob)) # ổn định tính toán số
            prob = tuple(p/sum(prob) for p in prob)
            
            current_dist = stats.rv_discrete(values=(list(range(len(prob))), prob))
            z_new = current_dist.rvs(size=1)
            z_new = int(z_new)

            if z_new == len(prob) - 1:
                clusters.append(Model(D=D))
                n_points_cluster.append(0)

            clusters[z_new].add_point(x_i)
            n_points_cluster[z_new] += 1
            z[i] = z_new
    K = len(clusters)
    return {'K': K, 'label': list(z)}

def DPMM_2(X, Model=PymcMultivariateNormal, K=1, z_init=None, alpha=1.0, max_iters=100):
    # transfer 2-dimension list X to numpy array
    for i in range(len(X)):
        X[i] = np.array(X[i])
    X = np.array(X)
    N = len(X)
    D = X.shape[1]

    if z_init is None:
        z = np.zeros(shape=N)
        K = len(np.unique(z))
    else:
        z = z_init
        K = len(np.unique(z))
    n_points_cluster = [0] * K

    clusters = [Model(D, X) for _ in range(K)]

    clusters[0].fitting()
    n_points_cluster[0] = N

    for _ in trange(max_iters):
        for i in range(N):
            z_i = z[i]
            x_i = X[i].reshape(D, 1)

            current_clus = clusters[z_i]

            current_clus.del_point(x_i)
            n_points_cluster[z_i] -= 1

            if n_points_cluster[z_i] == 0:
                del n_points_cluster[z_i]
                del clusters[z_i]
                z[z > z_i] -= 1
            prob = np.log(np.array(n_points_cluster + [alpha]))
            for j in range(len(clusters)):
                print("clusters[j].logpredictive(x_i): ", clusters[j].predict(x_i))
                prob[j] = prob[j] + clusters[j].predict(x_i)
            prob[-1] = prob[-1] + dummy_dist.predict(x_i)
            prob = np.exp(prob - np.max(prob)) # ổn định tính toán số
            prob = tuple(p/sum(prob) for p in prob)

            current_dist = stats.rv_discrete(values=(list(range(len(prob))), prob))
            z_new = current_dist.rvs(size=1)
            z_new = int(z_new)

            if z_new == len(prob) - 1:
                clusters.append(Model(D, np.array([x_i])))
                n_points_cluster.append(0)

            clusters[z_new].add_point(x_i)
            n_points_cluster[z_new] += 1
            z[i] = z_new
    K = len(clusters)
    return {'K': K, 'label': list(z)}

def generate_data():
    s0 = 6.0 # standard deviation between centers of clusters
    ss = 0.65 # Standard deviation between points in the cluster

    mean_1 = multivariate_normal.rvs(mean=np.array([0,0,0,0,0,0])) * s0
    mean_2 = multivariate_normal.rvs(mean=np.array([0,0,0,0,0,0])) * s0
    mean_3 = multivariate_normal.rvs(mean=np.array([0,0,0,0,0,0])) * s0
    mean_4 = multivariate_normal.rvs(mean=np.array([0,0,0,0,0,0])) * s0
    mean_5 = multivariate_normal.rvs(mean=np.array([0,0,0,0,0,0])) * s0

    X1 = mean_1 + multivariate_normal.rvs(mean=np.array([0,0,0,0,0,0]), size=100) * ss
    X2 = mean_2 + multivariate_normal.rvs(mean=np.array([0,0,0,0,0,0]), size=60) * ss
    X3 = mean_3 + multivariate_normal.rvs(mean=np.array([0,0,0,0,0,0]), size=40) * ss
    X4 = mean_4 + multivariate_normal.rvs(mean=np.array([0,0,0,0,0,0]), size=65) * ss
    X5 = mean_5 + multivariate_normal.rvs(mean=np.array([0,0,0,0,0,0]), size=80) * ss

    X = np.concatenate((X1, X2, X3))
    return X

def generate_data_two(num_clusters=4, points_per_cluster=40):
    data = []
    for i in range(num_clusters):
        mean = np.random.normal(0, 3, 2)
        data.append(mean + np.random.normal(0, 0.65, (points_per_cluster, 2)))
    return np.concatenate(data)

def generate_data_3():
    # Generate six-dimensional binary data containing only 0s and 1s.
    data = np.random.randint(0, 2, (100, 6))
    return data
