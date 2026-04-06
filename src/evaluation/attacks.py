import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator

class PrivacyAttacks:
    def __init__(self, real, syn, control, n_attacks=50):
        self.real = real
        self.syn = syn
        self.control = control
        # Safety check for sample sizes [cite: 304]
        self.n_attacks = min(n_attacks, len(real), len(control))

    # 1. Standard Attack: Singling Out [cite: 917]
    def singling_out(self):
        print("Running Singling Out Attack...")
        evaluator = SinglingOutEvaluator(ori=self.real, syn=self.syn, control=self.control, n_attacks=self.n_attacks)
        evaluator.evaluate()
        return evaluator.risk()

    # 2. Standard Attack: Linkability [cite: 918]
    def linkability(self):
        print("Running Linkability Attack...")
        aux_cols = list(self.real.columns)
        evaluator = LinkabilityEvaluator(ori=self.real, syn=self.syn, control=self.control, aux_cols=aux_cols, n_attacks=self.n_attacks)
        evaluator.evaluate()
        return evaluator.risk()

    # 3. CUSTOM RESEARCH ATTACK: CMLA (Based on Paper-1)
    def cmla_leakage(self, n_clusters=10):
        print("Running Custom Cluster-Medoid Leakage Attack...")
        # Use only numeric data for distance calculations [cite: 313]
        syn_numeric = self.syn.select_dtypes(include=[np.number]).values
        real_numeric = self.real.select_dtypes(include=[np.number]).values
        
        # Clustering to find 'neighborhoods'
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(syn_numeric)
        
        medoids = []
        for i in range(n_clusters):
            cluster_points = syn_numeric[kmeans.labels_ == i]
            if len(cluster_points) > 0:
                mean_v = cluster_points.mean(axis=0).reshape(1, -1)
                closest_idx, _ = pairwise_distances_argmin_min(mean_v, cluster_points)
                medoids.append(cluster_points[closest_idx[0]])
        
        # Calculate distance to nearest real records
        _, min_dist = pairwise_distances_argmin_min(np.array(medoids), real_numeric)
        
        # Return Attack Success Rate (ASR)
        asr = np.sum(min_dist < 0.05) / len(medoids)
        return asr