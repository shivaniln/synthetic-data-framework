import os
import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.models.copula_model import CopulaModel
from src.models.ctgan_model import CTGANModel
from src.models.tvae_model import TVAEModel
from src.evaluation.attacks import PrivacyAttacks
from src.evaluation.metrics import StatisticalMetrics
from src.utils.visualizer import Visualizer

def run_framework(filename):
    # 1. Setup
    print(f"🚀 Loading and cleaning: {filename}...")
    input_path = os.path.join('data', 'input', filename)
    loader = DataLoader(input_path)
    real_df, metadata = loader.load_and_clean()
    
    
    if len(real_df) > 2000:
        print(f"⚠️ Dataset too large ({len(real_df)} rows). Subsampling to 2000 for efficiency.")
        real_df = real_df.sample(n=2000, random_state=42).reset_index(drop=True)
    
    half = len(real_df) // 2
    train_df = real_df.iloc[:half]
    control_df = real_df.iloc[half:]
    
    viz = Visualizer()
    metrics = StatisticalMetrics()
    results = []
    generated_datasets = {} 

    # --- LEAD'S FIX: REDUCED EPOCHS FOR DEMO ---
    # 50 epochs is enough to show learning without taking 20 minutes
    models = {
        "Copula": CopulaModel(metadata),
        "CTGAN": CTGANModel(metadata, epochs=50), 
        "TVAE": TVAEModel(metadata, epochs=50)
    }

    # 2. Benchmarking Loop
    for name, model_obj in models.items():
        print(f"\n--- 🛠️ Auditing {name} ---")
        print(f"Training started (this may take a minute for GANs)...")
        model_obj.train(train_df)
        
        print(f"Generating synthetic data for {name}...")
        syn_df = model_obj.generate(len(train_df))
        generated_datasets[name] = syn_df
        
        print(f"Running Privacy & Logic Audits...")
        attacker = PrivacyAttacks(train_df, syn_df, control_df, n_attacks=20) # n_attacks lowered for speed
        s_risk = attacker.singling_out().value 
        l_risk = attacker.linkability().value    
        cmla_score = attacker.cmla_leakage()     
        
        logic_score, _ = metrics.logic_check(train_df, syn_df)
        real_corr = train_df.corr(numeric_only=True)
        syn_corr = syn_df.corr(numeric_only=True)
        c_error = np.abs(real_corr - syn_corr).mean().mean()
        
        results.append({
            "Model": name,
            "Singling_Out_Risk": s_risk,
            "Linkability_Risk": l_risk,
            "CMLA_Leakage": cmla_score,
            "Logic_Consistency": logic_score,
            "Utility_Error": c_error
        })

    # 3. Final Decision & Export Logic
    report_df = pd.DataFrame(results)
    
    # Selection Thresholds (Strict for high-quality research)
    safe_models = report_df[(report_df['CMLA_Leakage'] <= 0.10) & 
                            (report_df['Logic_Consistency'] >= 0.70)]

    print("\n" + "="*30)
    print("📋 FINAL RESEARCH VERDICT")
    print("="*30)
    
    if not safe_models.empty:
        best_row = safe_models.loc[safe_models['Utility_Error'].idxmin()]
        best_name = best_row['Model']
        print(f"🏆 RECOMMENDED METHOD: {best_name}")
        
        winner_df = generated_datasets[best_name]
        winner_filename = f"{best_name}_best_synthetic.csv"
        winner_path = os.path.join('data', 'output', winner_filename)
        winner_df.to_csv(winner_path, index=False)
        
        viz.plot_winner_comparison(train_df, winner_df, best_name)
        
        report_df['Is_Recommended'] = report_df['Model'] == best_name
        report_df.to_csv("data/output/final_audit_report.csv", index=False)
        print(f"✅ Success! Winner saved as: {winner_filename}")
    else:
        print("❌ CRITICAL: No model met the safety/logic thresholds.")
        print(report_df)

if __name__ == "__main__":
    os.makedirs(os.path.join('data', 'output'), exist_ok=True)
    # Ensure you change this to the name of your Adult Census file if it's different!
    run_framework("test.csv")