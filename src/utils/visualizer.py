import matplotlib.pyplot as plt
import seaborn as sns
import os

class Visualizer:
    def __init__(self, save_dir='data/output/'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_correlations(self, real_df, syn_df, name):
        plt.figure(figsize=(10, 8))
        diff = real_df.corr(numeric_only=True) - syn_df.corr(numeric_only=True)
        sns.heatmap(diff, annot=True, cmap='RdBu', center=0)
        plt.title(f'Correlation Difference: {name}')
        plt.savefig(os.path.join(self.save_dir, f'corr_diff_{name.lower()}.png'))
        plt.close()

    def plot_winner_comparison(self, real_df, best_syn_df, best_name):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.heatmap(real_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='Blues', ax=ax1)
        ax1.set_title('Original Data Correlations')
        sns.heatmap(best_syn_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='Greens', ax=ax2)
        ax2.set_title(f'Winner: {best_name} Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'winner_comparison.png'))
        plt.close()