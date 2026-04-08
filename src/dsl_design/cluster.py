import numpy as np
from tqdm import tqdm
import math

from utils.util import normalized_sampling
from sklearn.preprocessing import StandardScaler

class DPMM:
    def __init__(self):
        pass

    # def cluster(data, Distribution, feature_dim, iter_times=1000, alpha=0.3, regular=0.1, n_components=2, num_classes=2):
    #     Cluster = Distribution
    #     N = len(data)
    #     if N == 0: 
    #         return {
    #             "K": 0,
    #             "label": [],
    #             "log_likelihood_list": " ".join(["0" for _ in range(iter_times)])
    #         }
    #     data = np.array(data)
    #     label = np.zeros(len(data))
    #     cluster = Cluster(feature_dim, n_components=n_components, num_classes=num_classes)
    #     cluster.set_data_point(data, regular=regular)
    #     clusters = [cluster]
    #     log_likelihood_list = []

    #     # init with all data in one cluster
    #     for _ in tqdm(range(iter_times), desc="Processing", unit="iteration", leave=False):
    #         # Update data's cluster
    #         K = len(clusters)

    #         log_likelihood = 1
    #         for i in range(N):
    #             probs = np.zeros(K+1)
    #             for k in range(K):
    #                 probs[k] = clusters[k].probability(data[i]) * (clusters[k].N / (N - 1 + alpha))
    #             probs[K] = alpha / (N - 1 + alpha)
    #             new_label, normalized_data = normalized_sampling(probs)
    #             log_likelihood = log_likelihood * normalized_data[int(label[i])]
    #             label[i] = new_label
    #             if log_likelihood <= 0:
    #                 log_likelihood_list.append(log_likelihood_list[-1])
    #             else:
    #                 log_likelihood_list.append(math.log(log_likelihood)/N)
            
    #         # Update cluster's param
    #         k, clusters = 0, []
    #         for i in range(K+1):
    #             data_point = [x for x in range(N) if label[x] == i]
    #             if len(data_point) > 0:
    #                 cluster = Cluster(feature_dim)
    #                 cluster.set_data_point(data[data_point], regular=regular)
    #                 clusters.append(cluster)
    #                 label[data_point] = k
    #                 k += 1
    #     result = {
    #         "K": len(clusters),
    #         "label": label,
    #         "log_likelihood_list": " ".join([str(num) for num in log_likelihood_list])
    #     }
    #     # print("cluster result[K]: ", result["K"])
    #     return result

    def cluster(data, Distribution, feature_dim, iter_times=1000, alpha=0.3, regular=0.1, n_components=2, num_classes=2):
        Cluster = Distribution
        N = len(data)
        if N == 0: 
            return {
                "K": 0,
                "label": [],
                "log_likelihood_list": " ".join(["0" for _ in range(iter_times)])
            }

        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        data = np.array(data)
        label = np.zeros(len(data))
        cluster = Cluster(feature_dim, n_components=n_components, num_classes=num_classes)
        cluster.set_data_point(data, regular=regular)
        clusters = [cluster]
        log_likelihood_list = []

        # Initialize with all data in one cluster
        for _ in tqdm(range(iter_times), desc="Processing", unit="iteration", leave=False):
            # Step 1: Update data's cluster
            K = len(clusters)

            for i in range(N):
                probs = np.zeros(K + 1)
                for k in range(K):
                    # Calculate the probability of data[i] belonging to existing clusters
                    probs[k] = clusters[k].probability(data[i]) * (clusters[k].N / (N - 1 + alpha))
                # Calculate the probability of data[i] forming a new cluster
                probs[K] = alpha / (N - 1 + alpha)
                # Sample a new label for data[i]
                new_label, _ = normalized_sampling(probs)
                label[i] = new_label

            # Step 2: Update cluster parameters
            new_clusters = []
            k = 0
            for i in range(K + 1):
                data_point_indices = [x for x in range(N) if label[x] == i]
                if len(data_point_indices) > 0:
                    new_cluster = Cluster(feature_dim)
                    new_cluster.set_data_point(data[data_point_indices], regular=regular)
                    new_clusters.append(new_cluster)
                    # Update label to ensure consistency with new cluster indices
                    label[data_point_indices] = k
                    k += 1
            clusters = new_clusters

            # Step 3: Calculate and record log-likelihood
            log_likelihood = 0
            total_ll = 0
            for i in range(N):
                prob_sum = 0
                for k in range(len(clusters)):
                    prob_sum += clusters[k].probability(data[i]) * (clusters[k].N / (N + alpha))
                total_ll += np.log(prob_sum + 1e-12)

            # 记录平均对数似然
            avg_ll = total_ll / N
            log_likelihood_list.append(avg_ll)

        result = {
            "K": len(clusters),
            "label": label,
            "log_likelihood_list": " ".join([str(num) for num in log_likelihood_list])
        }
        return result
