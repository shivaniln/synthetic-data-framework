import os
import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.models.copula_model import CopulaModel
from src.models.ctgan_model import CTGANModel
from src.models.tvae_model import TVAEModel
from src.evaluation.attacks import PrivacyAttacks
from src.utils.visualizer import Visualizer

def run_framework(filename):
    # 1. Setup
    input_path = os.path.join('data', 'input', filename)
    loader = DataLoader(input_path)
    real_df, metadata = loader.load_and_clean()
    
    half = len(real_df) // 2
    train_df = real_df.iloc[:half]
    control_df = real_df.iloc[half:]
    
    viz = Visualizer()
    results = []
    generated_datasets = {}

    models = {
        "Copula": CopulaModel(metadata),
        "CTGAN": CTGANModel(metadata, epochs=20),
        "TVAE": TVAEModel(metadata, epochs=20)
    }

    # 2. Benchmarking Loop
    for name, model_obj in models.items():
        print(f"--- Benchmarking {name} ---")
        model_obj.train(train_df)
        syn_df = model_obj.generate(len(train_df))
        generated_datasets[name] = syn_df
        
        # Temporary save
        temp_path = os.path.join('data', 'output', f"synthetic_{name.lower()}.csv")
        syn_df.to_csv(temp_path, index=False)
        
        # Privacy Audit
        attacker = PrivacyAttacks(train_df, syn_df, control_df, n_attacks=50)
        l_risk = attacker.linkability().value
        s_risk = attacker.singling_out().value
        
        # Utility calculation (Correlation Error)
        real_corr = train_df.corr(numeric_only=True)
        syn_corr = syn_df.corr(numeric_only=True)
        c_error = np.abs(real_corr - syn_corr).mean().mean()
        
        results.append({
            "Model": name,
            "Linkability Risk": l_risk,
            "Singling Out Risk": s_risk,
            "Utility Error": c_error
        })

    # 3. Decision Logic & Cleanup
    report_df = pd.DataFrame(results)
    threshold = 0.05 # 5% max risk
    safe_models = report_df[(report_df['Linkability Risk'] <= threshold) & 
                            (report_df['Singling Out Risk'] <= threshold)]

    print("\n--- FINAL RESEARCH VERDICT ---")
    if not safe_models.empty:
        best_row = safe_models.loc[safe_models['Utility Error'].idxmin()]
        best_name = best_row['Model']
        print(f"🏆 RECOMMENDED METHOD: {best_name}")
        
        # Clean up files of losing models
        for name in models.keys():
            if name != best_name:
                file_path = os.path.join('data', 'output', f"synthetic_{name.lower()}.csv")
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Final visual and report
        viz.plot_winner_comparison(train_df, generated_datasets[best_name], best_name)
        report_df['Is_Recommended'] = report_df['Model'] == best_name
        report_df.to_csv("data/output/final_report.csv", index=False)
        print(f"Results finalized in data/output/")
    else:
        print("No model met safety requirements.")

if __name__ == "__main__":
    os.makedirs(os.path.join('data', 'output'), exist_ok=True)
    run_framework("test.csv")