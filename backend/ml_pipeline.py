import os
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json
import gdown
from scipy.stats import poisson
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive sessions (Jupyter, python -c, etc.)
    BASE_DIR = os.path.abspath(os.getcwd())
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Point to data dir in backend/data
DATA_DIR = os.path.join(BASE_DIR, "data")

# =====================================================================
# DATA DOWNLOAD  (Google Drive folder → Dropbox zip → Google Drive per-file)
# =====================================================================

# Shared Google Drive folder URL (publicly accessible via link)
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1r6tz8ySbS6_sB-OFiN-C1-iPuv6icKfc?usp=sharing"

# ---- Per-file IDs (resolved from the shared folder above) ----
# Used as fallback if download_folder fails.
# Refresh IDs: python -c "import gdown; gdown.download_folder(id='1r6tz8ySbS6_sB-OFiN-C1-iPuv6icKfc', skip_download=True)"
DRIVE_FILE_IDS = {
    "PSL_xG_Database_2425.csv":     "1WD-1EqrT-JG9zTwyNTAKa4BI3BF909pW",
    "PSL_xG_Database_2526.csv":     "1302MIif0X0SbCl2EJzN8fsU_l1davr98",
    "PSL_MatchStats_2425.csv":      "13h_gfu7NluMHVFvAhxpbehbC9sHC2roJ",
    "PSL_MatchStats_2526.csv":      "168oQhLNZcxxytcqzVBCDjIxR8CZd7PNR",
    "PSL_Events_Database_2425.csv": "1_r896pPucCReSAoCuoU2HqcwM87Y5ff1",
    "PSL_Events_Database_2526.csv": "18Hy2zGL2xVFOdegInBVyc4M1wxb3XY3F",
}



def download_data_from_drive(force: bool = False):
    """
    Télécharge tous les CSV dans DATA_DIR via une stratégie à 3 niveaux :
      1. Google Drive  — dossier complet (gdown.download_folder)
      2. Dropbox       — dossier téléchargé en zip puis extrait
      3. Google Drive  — fichier par fichier (IDs individuels)
    force=True force le re-téléchargement même si les fichiers sont présents.
    """
    import urllib.request
    import zipfile
    import tempfile

    DROPBOX_ZIP_URL = (
        "https://www.dropbox.com/scl/fo/0t9ft4lckfe3th6ihpz0i/"
        "AP6cD8QVvwdrZR5sG9pF2cg"
        "?rlkey=ngia4vedhpm5y5ckidl3vwrqe&dl=1"
    )

    os.makedirs(DATA_DIR, exist_ok=True)

    def _all_present():
        return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in DRIVE_FILE_IDS)

    if _all_present() and not force:
        print("[Data] Tous les fichiers sont déjà présents, téléchargement ignoré.")
        return

    # ------------------------------------------------------------------
    # 1. Google Drive — dossier complet
    # ------------------------------------------------------------------
    print("[Drive] Tentative téléchargement dossier Google Drive...")
    try:
        gdown.download_folder(
            url=DRIVE_FOLDER_URL,
            output=DATA_DIR,
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
        if _all_present():
            print("[Drive] ✓ Téléchargement Google Drive terminé.")
            return
        print("[Drive] Fichiers manquants après download_folder, passage au plan B.")
    except Exception as e:
        print(f"[Drive] download_folder a échoué : {e}")

    # ------------------------------------------------------------------
    # 2. Dropbox — dossier entier zippé → extraction des CSV
    # ------------------------------------------------------------------
    print("[Dropbox] Tentative téléchargement dossier Dropbox (zip)...")
    try:
        tmp_zip = tempfile.mktemp(suffix=".zip")
        req = urllib.request.Request(DROPBOX_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_zip, "wb") as out:
            out.write(resp.read())
        print("[Dropbox] Archive reçue, extraction en cours...")
        with zipfile.ZipFile(tmp_zip, "r") as z:
            for member in z.namelist():
                fname = os.path.basename(member)
                if fname in DRIVE_FILE_IDS:
                    dest = os.path.join(DATA_DIR, fname)
                    with z.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    print(f"[Dropbox]   ✓ {fname}")
        os.remove(tmp_zip)
        if _all_present():
            print("[Dropbox] ✓ Tous les fichiers extraits avec succès.")
            return
        print("[Dropbox] Extraction partielle, passage au plan C.")
    except Exception as e:
        print(f"[Dropbox] Échec : {e}")

    # ------------------------------------------------------------------
    # 3. Google Drive — fichier par fichier (fallback final)
    # ------------------------------------------------------------------
    print("[Drive] Téléchargement fichier par fichier (fallback final)...")
    for filename, file_id in DRIVE_FILE_IDS.items():
        dest = os.path.join(DATA_DIR, filename)
        if not os.path.exists(dest) or force:
            print(f"[Drive] Téléchargement de {filename}...")
            try:
                gdown.download(
                    f"https://drive.google.com/uc?id={file_id}",
                    dest,
                    quiet=False,
                )
                print(f"[Drive]   ✓ {filename}")
            except Exception as e:
                print(f"[Drive]   ✗ Échec pour {filename} : {e}")
        else:
            print(f"[Drive]   — {filename} déjà présent, ignoré.")


# =====================================================================
# DATA LOADING & CONSOLIDATION
# =====================================================================

def load_and_combine_seasons(file_prefix, seasons=['2425', '2526']):
    dfs = []
    for season in seasons:
        file_path = os.path.join(DATA_DIR, f"{file_prefix}_{season}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Season'] = season
            dfs.append(df)
        else:
            print(f"Warning: {file_path} not found. Skipping.")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def parse_match_data(xg_df):
    matches = xg_df[['Date', 'Game', 'Season']].drop_duplicates().sort_values('Date').reset_index(drop=True)
    matches['Date'] = pd.to_datetime(matches['Date'])

    parsed_data = []
    for _, row in matches.iterrows():
        try:
            teams_part, score_part = row['Game'].split(' - ')
            home_team, away_team = teams_part.split(' vs ')
            home_goals, away_goals = map(int, score_part.split(':'))
            parsed_data.append([row['Date'], row['Season'], row['Game'], home_team.strip(), away_team.strip(), home_goals, away_goals])
        except ValueError:
            pass

    df_matches = pd.DataFrame(parsed_data, columns=['Date', 'Season', 'Game', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals'])
    conditions = [
        (df_matches['HomeGoals'] > df_matches['AwayGoals']),
        (df_matches['HomeGoals'] == df_matches['AwayGoals']),
        (df_matches['HomeGoals'] < df_matches['AwayGoals'])
    ]
    df_matches['Target'] = np.select(conditions, [2, 1, 0], default=1)
    return df_matches

def aggregate_team_stats(df_matches, xg_df, stats_df, events_df):
    team_xg = xg_df.groupby(['Game', 'Team'])[['xG', 'xGOT']].sum().reset_index()

    team_stats = pd.DataFrame()
    if not stats_df.empty:
        team_col = 'Team' if 'Team' in stats_df.columns else 'teamName' if 'teamName' in stats_df.columns else 'contestantId'
        stat_cols = ['totalPass', 'totalFinalThirdPasses', 'duelWon', 'touchesInOppBox', 'interceptionWon']
        available_stats = [c for c in stat_cols if c in stats_df.columns]
        if available_stats:
            team_stats = stats_df.groupby(['Game', team_col])[available_stats].sum().reset_index()
            team_stats.rename(columns={team_col: 'Team'}, inplace=True)

    team_events = pd.DataFrame()
    if not events_df.empty and 'xT' in events_df.columns:
        team_col = 'Team' if 'Team' in events_df.columns else 'teamName'
        team_events = events_df.groupby(['Game', team_col])['xT'].sum().reset_index()
        team_events.rename(columns={team_col: 'Team'}, inplace=True)

    def merge_team_features(df, agg_df, prefix):
        if agg_df.empty: return df
        df = df.merge(agg_df, left_on=['Game', 'HomeTeam'], right_on=['Game', 'Team'], how='left')
        df = df.drop('Team', axis=1).rename(columns={c: f'Home_{c}' for c in agg_df.columns if c not in ['Game', 'Team']})
        df = df.merge(agg_df, left_on=['Game', 'AwayTeam'], right_on=['Game', 'Team'], how='left')
        df = df.drop('Team', axis=1).rename(columns={c: f'Away_{c}' for c in agg_df.columns if c not in ['Game', 'Team']})
        return df

    df_matches = merge_team_features(df_matches, team_xg, '')
    df_matches = merge_team_features(df_matches, team_stats, '')
    df_matches = merge_team_features(df_matches, team_events, '')

    df_matches.fillna(0, inplace=True)
    return df_matches

def calculate_elo_ratings(df_matches, k_factor=20):
    elo_dict = {}
    home_elos, away_elos = [], []
    for index, row in df_matches.iterrows():
        home_team, away_team = row['HomeTeam'], row['AwayTeam']
        home_elo = elo_dict.get(home_team, 1500.0)
        away_elo = elo_dict.get(away_team, 1500.0)
        home_elos.append(home_elo)
        away_elos.append(away_elo)

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home
        actual_home = 1 if row['Target'] == 2 else (0 if row['Target'] == 0 else 0.5)
        actual_away = 1 - actual_home

        elo_dict[home_team] = home_elo + k_factor * (actual_home - expected_home)
        elo_dict[away_team] = away_elo + k_factor * (actual_away - expected_away)

    df_matches['Home_Elo'] = home_elos
    df_matches['Away_Elo'] = away_elos
    df_matches['Elo_Difference'] = df_matches['Home_Elo'] - df_matches['Away_Elo']
    
    # Save the final ELO for predictions
    joblib.dump(elo_dict, os.path.join(MODEL_DIR, 'latest_elo.joblib'))
    return df_matches

def build_rolling_features(df_matches):
    metric_cols = [c.replace('Home_', '') for c in df_matches.columns if c.startswith('Home_') and c not in ['HomeTeam', 'HomeGoals', 'Home_Elo']]
    team_matches = []
    for _, row in df_matches.iterrows():
        home_data = {'Date': row['Date'], 'Game': row['Game'], 'Team': row['HomeTeam']}
        for col in metric_cols:
            home_data[f'{col}_For'] = row.get(f'Home_{col}', 0)
            home_data[f'{col}_Against'] = row.get(f'Away_{col}', 0)
        team_matches.append(home_data)

        away_data = {'Date': row['Date'], 'Game': row['Game'], 'Team': row['AwayTeam']}
        for col in metric_cols:
            away_data[f'{col}_For'] = row.get(f'Away_{col}', 0)
            away_data[f'{col}_Against'] = row.get(f'Home_{col}', 0)
        team_matches.append(away_data)

    df_teams = pd.DataFrame(team_matches).sort_values(['Team', 'Date'])
    window = 5
    rolling_cols = []
    for col in metric_cols:
        df_teams[f'Roll_{col}_For'] = df_teams.groupby('Team')[f'{col}_For'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        df_teams[f'Roll_{col}_Against'] = df_teams.groupby('Team')[f'{col}_Against'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        rolling_cols.extend([f'Roll_{col}_For', f'Roll_{col}_Against'])

    df_teams['Rest_Days'] = df_teams.groupby('Team')['Date'].diff().dt.days.fillna(14).clip(upper=21)
    rolling_cols.append('Rest_Days')

    for is_home, prefix in [(True, 'Home_'), (False, 'Away_')]:
        team_type = 'HomeTeam' if is_home else 'AwayTeam'
        merge_df = df_teams[['Game', 'Team'] + rolling_cols].rename(columns={c: f'{prefix}{c}' for c in rolling_cols})
        df_matches = df_matches.merge(merge_df, left_on=['Game', team_type], right_on=['Game', 'Team'], how='left').drop('Team', axis=1)

    df_matches.fillna(df_matches.mean(numeric_only=True), inplace=True)
    
    # Save the latest rolling stats per team
    latest_stats = df_teams.groupby('Team').last()[rolling_cols].to_dict('index')
    joblib.dump(latest_stats, os.path.join(MODEL_DIR, 'latest_rolling_stats.joblib'))
    
    return df_matches

def add_poisson_probabilities(df_matches):
    def get_match_probs(h_lambda, a_lambda):
        h_lambda, a_lambda = max(0.1, h_lambda), max(0.1, a_lambda)
        h_probs = [poisson.pmf(i, h_lambda) for i in range(7)]
        a_probs = [poisson.pmf(i, a_lambda) for i in range(7)]
        matrix = np.outer(h_probs, a_probs)
        return np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

    probs = []
    for _, row in df_matches.iterrows():
        if 'Home_Roll_xG_For' in row:
            h_proj = (row['Home_Roll_xG_For'] + row['Away_Roll_xG_Against']) / 2
            a_proj = (row['Away_Roll_xG_For'] + row['Home_Roll_xG_Against']) / 2
        else:
            h_proj, a_proj = 1.0, 1.0
        probs.append(get_match_probs(h_proj, a_proj))

    prob_df = pd.DataFrame(probs, columns=['Poisson_HomeWin', 'Poisson_Draw', 'Poisson_AwayWin'])
    totals = prob_df.sum(axis=1)
    for col in prob_df.columns: prob_df[col] /= totals

    return pd.concat([df_matches, prob_df], axis=1)

# =====================================================================
# MAIN PIPELINE FUNCTIONS
# =====================================================================

def trigger_training():
    # Téléchargement automatique depuis Google Drive (ignoré si déjà présent)
    download_data_from_drive()

    xg_df = load_and_combine_seasons("PSL_xG_Database")
    stats_df = load_and_combine_seasons("PSL_MatchStats")
    events_df = load_and_combine_seasons("PSL_Events_Database")

    if xg_df.empty:
        raise ValueError("xG Database is missing or empty. Cannot train.")

    df = parse_match_data(xg_df)
    df = aggregate_team_stats(df, xg_df, stats_df, events_df)
    df = calculate_elo_ratings(df)
    df = build_rolling_features(df)
    df = add_poisson_probabilities(df)

    # Missing default value fill mechanism
    features_ordered = [c for c in df.columns if c.startswith(('Home_Roll_', 'Away_Roll_', 'Poisson_', 'Elo_')) or c in ['Home_Elo', 'Away_Elo', 'Home_Rest_Days', 'Away_Rest_Days']]
    df[features_ordered] = df[features_ordered].fillna(0)

    X = df[features_ordered]
    y = df['Target']
    
    # Save the current list of expected features
    joblib.dump(features_ordered, os.path.join(MODEL_DIR, 'features_list.joblib'))
    joblib.dump(list(df['HomeTeam'].unique()), os.path.join(MODEL_DIR, 'teams.joblib'))

    tscv = TimeSeriesSplit(n_splits=3) # kept small for speed
    sample_weights = compute_sample_weight('balanced', y)

    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'random_state': 42
        }
        log_losses = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights[train_idx], verbose=False)
            preds = model.predict_proba(X_val)
            log_losses.append(log_loss(y_val, preds, labels=[0,1,2]))
        return np.mean(log_losses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) # 10 trials for quick dev
    
    best_params = study.best_params
    best_params['objective'] = 'multi:softprob'
    best_params['num_class'] = 3
    best_params['random_state'] = 42

    base_model = xgb.XGBClassifier(**best_params)
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=tscv)
    calibrated_model.fit(X, y, sample_weight=sample_weights)

    joblib.dump(calibrated_model, os.path.join(MODEL_DIR, 'calibrated_model.joblib'))

    # SHAP Generation
    final_explainer = xgb.XGBClassifier(**best_params)
    final_explainer.fit(X, y, sample_weight=sample_weights)
    explainer = shap.TreeExplainer(final_explainer)
    shap_vals = explainer.shap_values(X)
    
    # Store aggregated SHAP for insights api
    feature_importances = np.abs(shap_vals).mean(axis=(0, 1)) if shap_vals[0].ndim == 2 else np.abs(shap_vals).mean(axis=0)
    shap_dict = dict(zip(features_ordered, feature_importances.tolist()))
    shap_dict = dict(sorted(shap_dict.items(), key=lambda item: item[1], reverse=True)[:15]) # top 15
    
    with open(os.path.join(MODEL_DIR, 'shap_data.json'), 'w') as f:
        json.dump(shap_dict, f)

    return {"status": "success", "message": "Model trained and saved."}

def load_cached_predictor():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'calibrated_model.joblib'))
        features_list = joblib.load(os.path.join(MODEL_DIR, 'features_list.joblib'))
        latest_elo = joblib.load(os.path.join(MODEL_DIR, 'latest_elo.joblib'))
        latest_stats = joblib.load(os.path.join(MODEL_DIR, 'latest_rolling_stats.joblib'))
        return model, features_list, latest_elo, latest_stats
    except Exception as e:
        return None, None, None, None

def get_teams():
    try:
        return sorted(joblib.load(os.path.join(MODEL_DIR, 'teams.joblib')))
    except:
        return []

def get_shap():
    try:
        with open(os.path.join(MODEL_DIR, 'shap_data.json'), 'r') as f:
            return json.load(f)
    except:
        return {}

def predict_single_match(home_team, away_team):
    model, features_list, latest_elo, latest_stats = load_cached_predictor()
    if not model:
        raise ValueError("Model is not trained yet.")

    home_elo = latest_elo.get(home_team, 1500)
    away_elo = latest_elo.get(away_team, 1500)
    
    h_stats = latest_stats.get(home_team, {})
    a_stats = latest_stats.get(away_team, {})

    row = {}
    row['Home_Elo'] = home_elo
    row['Away_Elo'] = away_elo
    row['Elo_Difference'] = home_elo - away_elo
    
    # Map stats
    for k, v in h_stats.items(): row[f"Home_{k}"] = v
    for k, v in a_stats.items(): row[f"Away_{k}"] = v

    # Poisson
    h_proj = (row.get('Home_Roll_xG_For', 1) + row.get('Away_Roll_xG_Against', 1)) / 2
    a_proj = (row.get('Away_Roll_xG_For', 1) + row.get('Home_Roll_xG_Against', 1)) / 2

    def get_probs(h_lambda, a_lambda):
        h_probs = [poisson.pmf(i, max(0.1, h_lambda)) for i in range(7)]
        a_probs = [poisson.pmf(i, max(0.1, a_lambda)) for i in range(7)]
        matrix = np.outer(h_probs, a_probs)
        return np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

    h_win, draw, a_win = get_probs(h_proj, a_proj)
    row['Poisson_HomeWin'] = h_win
    row['Poisson_Draw'] = draw
    row['Poisson_AwayWin'] = a_win

    # Default missing features
    input_vector = []
    for f in features_list:
        input_vector.append(row.get(f, 0))

    X_pred = pd.DataFrame([input_vector], columns=features_list)
    probs = model.predict_proba(X_pred)[0]
    
    classes = ['Away Win', 'Draw', 'Home Win']
    pred_idx = int(np.argmax(probs))
    
    return {
        "home_win_prob": round(probs[2] * 100, 2),
        "draw_prob": round(probs[1] * 100, 2),
        "away_win_prob": round(probs[0] * 100, 2),
        "prediction": classes[pred_idx]
    }
