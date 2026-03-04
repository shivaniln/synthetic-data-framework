import pandas as pd
import numpy as np

class StatisticalMetrics:
    def correlation_diff(self, real_df, syn_df):
        # Calculate correlation matrices for numeric data
        real_corr = real_df.corr(numeric_only=True)
        syn_corr = syn_df.corr(numeric_only=True)
        
        # Compute the mean absolute difference
        diff = np.abs(real_corr - syn_corr)
        return diff.mean().mean()