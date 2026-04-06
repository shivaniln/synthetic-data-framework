import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

class Visualizer:
    def __init__(self, save_dir='data/output/'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # 1. Statistical Check: How well did the model learn the patterns?
    def plot_correlations(self, real_df, syn_df, name):
        plt.figure(figsize=(10, 8))
        # Calculates the difference between real and synthetic correlation matrices
        diff = real_df.corr(numeric_only=True) - syn_df.corr(numeric_only=True)
        sns.heatmap(diff, annot=True, cmap='RdBu', center=0)
        plt.title(f'Correlation Difference: {name}')
        plt.savefig(os.path.join(self.save_dir, f'corr_diff_{name.lower()}.png'))
        plt.close()

    # 2. Final Selection: The 'Judge's Dashboard'
    def plot_tradeoff_summary(self, audit_results_df):
        """
        Custom Contribution: Plots Privacy Risk vs. Utility Score
        audit_results_df should contain: Model_Name, Privacy_Risk, Utility_Score
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=audit_results_df, x='Privacy_Risk', y='Utility_Score', 
                        hue='Model_Name', style='Model_Name', s=200)
        
        plt.axvline(x=0.1, color='r', linestyle='--', label='High Risk Threshold')
        plt.title('Final Model Selection: Privacy vs. Utility')
        plt.xlabel('Privacy Risk (ASR/Linkability)')
        plt.ylabel('ML Utility (F1-Score)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'final_decision_dashboard.png'))
        plt.close()

    # 3. Winning Comparison: Show the guide the best results
    def plot_winner_comparison(self, real_df, best_syn_df, best_name):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(real_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='Blues', ax=ax1)
        ax1.set_title('Original Data Correlations')
        sns.heatmap(best_syn_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='Greens', ax=ax2)
        ax2.set_title(f'Winner: {best_name} Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'winner_comparison.png'))
        plt.close()