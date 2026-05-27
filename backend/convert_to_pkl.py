"""
Convertit les artefacts du modèle de .joblib → .pkl (pickle standard)
pour un déploiement léger sur Streamlit.
"""
import os
import pickle
import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models_pkl')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES_TO_CONVERT = [
    'calibrated_model.joblib',
    'features_list.joblib',
    'latest_elo.joblib',
    'latest_rolling_stats.joblib',
    'teams.joblib',
]

print(f"Source:  {MODEL_DIR}")
print(f"Output:  {os.path.abspath(OUTPUT_DIR)}")
print()

for fname in FILES_TO_CONVERT:
    src = os.path.join(MODEL_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, fname.replace('.joblib', '.pkl'))
    
    if not os.path.exists(src):
        print(f"  ✗ {fname} — INTROUVABLE, ignoré")
        continue
    
    data = joblib.load(src)
    with open(dst, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    src_size = os.path.getsize(src) / 1024
    dst_size = os.path.getsize(dst) / 1024
    print(f"  ✓ {fname:40s} → {os.path.basename(dst):40s}  ({src_size:.1f} KB → {dst_size:.1f} KB)")

# Copy shap_data.json as-is
import shutil
shap_src = os.path.join(MODEL_DIR, 'shap_data.json')
shap_dst = os.path.join(OUTPUT_DIR, 'shap_data.json')
if os.path.exists(shap_src):
    shutil.copy2(shap_src, shap_dst)
    print(f"  ✓ {'shap_data.json':40s} → {'shap_data.json':40s}  (copié tel quel)")

print()
print("Conversion terminée !")
