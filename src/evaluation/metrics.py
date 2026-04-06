import pandas as pd
import numpy as np

class StatisticalMetrics:
    def correlation_diff(self, real_df, syn_df):
        """
        Calculates the average difference in correlations between real and synthetic data.
        Lower error indicates higher statistical utility.
        """
        real_corr = real_df.corr(numeric_only=True)
        syn_corr = syn_df.corr(numeric_only=True)
        
        # Compute the mean absolute difference
        diff = np.abs(real_corr - syn_corr)
        return diff.mean().mean()

    def logic_check(self, real_df, syn_df):
        """
        GENERAL PURPOSE Logic Gatekeeper
        Automatically detects boundaries from the real data and ensures 
        the synthetic data stays within those logical 'physics'.
        """
        invalid_count = 0
        total_records = len(syn_df)

        # 1. Boundary Check: Ensure synthetic values are within real-world Min/Max
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            real_min = real_df[col].min()
            real_max = real_df[col].max()
            
            # Count rows that fall outside the observed physical range
            out_of_bounds = syn_df[(syn_df[col] < real_min) | (syn_df[col] > real_max)]
            invalid_count += len(out_of_bounds)

        # 2. Null-Invariant Check: If real was 100% full, synthetic shouldn't have NaNs
        for col in real_df.columns:
            if real_df[col].isnull().sum() == 0:
                if syn_df[col].isnull().sum() > 0:
                    invalid_count += syn_df[col].isnull().sum()

        # Calculate a General Consistency Score (0 to 1)
        logic_score = max(0, 1 - (invalid_count / (total_records * len(numeric_cols) if len(numeric_cols) > 0 else 1)))
        
        return logic_score, None