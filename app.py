import streamlit as st
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(page_title="LoL Esports Intel", page_icon="🎮", layout="centered")
st.title("🏆 LoL Pro Match Predictor")

# 2. Load and Cache Data
@st.cache_data(ttl=3600)
def load_data():
    conn = sqlite3.connect('league_database.sqlite')
    query = """
        SELECT 
            SUBSTR(date, 1, 10) as match_date, game_id, league, team_name, opponent_team_name as opponent,
            CASE WHEN result IN ('TRUE', 'True', '1') THEN 'Win' ELSE 'Loss' END as result,
            MAX(CAST(gamelength AS FLOAT)) / 60.0 AS duration_mins, SUM(CAST(kills AS FLOAT)) AS frags,
            MAX(CASE WHEN firstbloodkill IN ('TRUE', 'True', '1') THEN 1 ELSE 0 END) AS first_blood,
            SUM(CAST(killsat10 AS FLOAT)) AS early_frags
        FROM player_match_stats
        GROUP BY game_id, league, team_name, opponent_team_name, date, result
    """
    df_sql = pd.read_sql_query(query, conn)
    conn.close()

    try:
        csv_url = 'https://oracleselixir-downloadable-match-data.s3.us-west-2.amazonaws.com/2026_LoL_esports_match_data_from_OraclesElixir.csv'
        df_csv_raw = pd.read_csv(csv_url, low_memory=False)
        df_csv_team = df_csv_raw[df_csv_raw['position'] == 'team'].copy()
        
        if 'opponent' not in df_csv_team.columns:
            df_csv_team['opponent'] = df_csv_team.groupby('gameid')['teamname'].transform(
                lambda x: x.iloc[::-1].values if len(x) == 2 else [None] * len(x)
            )

        df_csv = pd.DataFrame({
            'match_date': pd.to_datetime(df_csv_team['date']).dt.strftime('%Y-%m-%d'),
            'game_id': df_csv_team['gameid'], 'league': df_csv_team['league'],
            'team_name': df_csv_team['teamname'], 'opponent': df_csv_team['opponent'],
            'result': df_csv_team['result'].apply(lambda x: 'Win' if str(x) in ['1', '1.0', 'True', 'TRUE'] else 'Loss'),
            'duration_mins': df_csv_team['gamelength'] / 60.0, 'frags': df_csv_team['kills'],
            'first_blood': df_csv_team['firstblood'].apply(lambda x: 1 if str(x) in ['1', '1.0', 'True', 'TRUE'] else 0),
            'early_frags': df_csv_team['killsat10'] if 'killsat10' in df_csv_team.columns else None
        })
    except Exception:
        df_csv = pd.DataFrame()

    master_df = pd.concat([df_sql, df_csv]).drop_duplicates(subset=['game_id', 'team_name']).dropna(subset=['team_name'])
    
    # 🔴 USER REQUESTED FILTER: LCK, LPL, LEC, and LCS ONLY 🔴
    major_leagues = ['LPL', 'LCK', 'LEC', 'LCS']
    master_df = master_df[master_df['league'].str.contains('|'.join(major_leagues), case=False, na=False)]
    
    master_df['match_date'] = pd.to_datetime(master_df['match_date'], errors='coerce')
    latest_db_date = master_df['match_date'].max()
    team_last_match = master_df.groupby('team_name')['match_date'].max()
    active_teams = team_last_match[team_last_match >= (latest_db_date - pd.Timedelta(days=120))].index
    master_df = master_df[master_df['team_name'].isin(active_teams)]
    
    return master_df

# 3. Train ML Model with Advanced Preprocessing
@st.cache_resource
def train_ml_model(df):
    ml_df = df.sort_values(by=['team_name', 'match_date', 'game_id'], ascending=True).copy()
    
    # Base Features
    ml_df['past_win_rate'] = ml_df.groupby('team_name')['result'].transform(lambda x: x.apply(lambda y: 1 if y == 'Win' else 0).shift(1).rolling(5, min_periods=1).mean())
    ml_df['past_avg_frags'] = ml_df.groupby('team_name')['frags'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    ml_df['past_early_frags'] = ml_df.groupby('team_name')['early_frags'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    # Opponent Strength Mapping
    opponent_stats = ml_df[['game_id', 'team_name', 'past_win_rate']].set_index(['game_id', 'team_name'])
    
    def get_opponent_win_rate(row):
        try:
            return opponent_stats.loc[(row['game_id'], row['opponent']), 'past_win_rate']
        except KeyError:
            return 0.5 
            
    ml_df['opp_past_win_rate'] = ml_df.apply(get_opponent_win_rate, axis=1)
    
    # The new massive feature: Strength Differential
    ml_df['strength_diff'] = ml_df['past_win_rate'] - ml_df['opp_past_win_rate']

    ml_df = ml_df.dropna(subset=['past_win_rate', 'past_avg_frags', 'opp_past_win_rate'])
    
    X = ml_df[['past_win_rate', 'strength_diff', 'past_avg_frags', 'past_early_frags']]
    y = ml_df['result'].apply(lambda x: 1 if x == 'Win' else 0)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    if not X.empty:
        model.fit(X, y)
        
    return model, ml_df

# Load & Train
with st.spinner('Downloading Live Data & Training Advanced ML Engine...'):
    master_df = load_data()
    ml_model, ml_features_df = train_ml_model(master_df)
    
    master_df = master_df.sort_values(by='match_date', ascending=False)
    available_teams = sorted(master_df['team_name'].unique())

# 4. User Interface
col1, col2 = st.columns(2)
with col1:
    primary_team = st.selectbox("Select Primary Team:", available_teams)
with col2:
    h2h_opponent = st.selectbox("Select H2H Opponent:", ["None"] + available_teams)

# 5. Search & Predict Logic
if st.button("Generate Intel & Prediction", use_container_width=True):
    team_df = master_df[master_df['team_name'] == primary_team].copy()
    team_df['win_val'] = (team_df['result'] == 'Win').astype(int)
    
    # Predict future win probability
    team_ml_history = ml_features_df[ml_features_df['team_name'] == primary_team]
    ml_prediction_text = "Not enough data for ML prediction."
    
    if not team_ml_history.empty:
        latest_stats = team_ml_history.iloc[-1]
        
        # Calculate context based on the H2H dropdown
        if h2h_opponent != "None":
            opp_history = ml_features_df[ml_features_df['team_name'] == h2h_opponent]
            current_opp_win_rate = opp_history.iloc[-1]['past_win_rate'] if not opp_history.empty else 0.5
            opp_name_display = h2h_opponent.upper()
        else:
            current_opp_win_rate = 0.5 # Baseline for a generic opponent
            opp_name_display = "AN AVERAGE OPPONENT"
            
        current_strength_diff = latest_stats['past_win_rate'] - current_opp_win_rate
        
        X_predict = [[
            latest_stats['past_win_rate'], 
            current_strength_diff, 
            latest_stats['past_avg_frags'], 
            latest_stats['past_early_frags']
        ]]
        
        win_prob = ml_model.predict_proba(X_predict)[0][1] * 100
        
        if win_prob >= 50:
            ml_prediction_text = f"🟢 FAVORED: {win_prob:.1f}% chance to win vs {opp_name_display}."
        else:
            ml_prediction_text = f"🔴 UNDERDOG: {win_prob:.1f}% chance to win vs {opp_name_display}."

    # Group games into Series
    series_df = team_df.groupby(['match_date', 'opponent']).agg(
        games_played=('game_id', 'count'), wins=('win_val', 'sum'),
        avg_frags=('frags', 'mean'), avg_early_frags=('early_frags', 'mean')
    ).reset_index()
    series_df['losses'] = series_df['games_played'] - series_df['wins']
    
    def get_outcome(row):
        if row['wins'] > row['losses']: return 'WIN'
        elif row['losses'] > row['wins']: return 'LOSS'
        else: return 'DRAW'
        
    series_df['outcome'] = series_df.apply(get_outcome, axis=1)
    series_df['score'] = series_df['wins'].astype(str) + "-" + series_df['losses'].astype(str)
    series_df = series_df.sort_values(by='match_date', ascending=False)
    
    out = []
    out.append(f"🤖 ADVANCED ML PREDICTION ENGINE")
    out.append(ml_prediction_text)
    out.append("\n" + "="*40 + "\n")
    
    # --- CALCULATE ADVANCED PROBABILITIES ---
    total_games = len(team_df)
    
    if total_games > 0:
        over_26_5 = len(team_df[team_df['frags'] > 26.5])
        over_pct = (over_26_5 / total_games) * 100
        under_pct = 100 - over_pct
        
        fb_games = team_df[team_df['first_blood'] == 1]
        fb_win_pct = (fb_games['win_val'].sum() / len(fb_games)) * 100 if len(fb_games) > 0 else 0
        
        no_fb_games = team_df[team_df['first_blood'] == 0]
        no_fb_win_pct = (no_fb_games['win_val'].sum() / len(no_fb_games)) * 100 if len(no_fb_games) > 0 else 0
        
        fast_games = team_df[team_df['duration_mins'] < 32.0]
        late_games = team_df[team_df['duration_mins'] >= 32.0]
        fast_win_pct = (fast_games['win_val'].sum() / len(fast_games)) * 100 if len(fast_games) > 0 else 0
        late_win_pct = (late_games['win_val'].sum() / len(late_games)) * 100 if len(late_games) > 0 else 0

        out.append(f"=== PROBABILITY & SUCCESS METRICS: {primary_team.upper()} ===")
        out.append(f"Match Record:      {team_df['win_val'].sum()}W - {total_games - team_df['win_val'].sum()}L")
        out.append(f"Frags Over 26.5:   {over_pct:.1f}%")
        out.append(f"Frags Under 26.5:  {under_pct:.1f}%")
        out.append(f"Win % (Got 1st Blood):  {fb_win_pct:.1f}%")
        out.append(f"Win % (Lost 1st Blood): {no_fb_win_pct:.1f}% (Comeback rate)")
        out.append(f"Win % (Fast <32m): {fast_win_pct:.1f}%")
        out.append(f"Win % (Late 32m+): {late_win_pct:.1f}%\n")
    else:
        out.append(f"Not enough historical data to calculate probabilities for {primary_team.upper()}.\n")
    
    out.append("=== RECENT 3 SERIES ===")
    for _, row in series_df.head(3).iterrows():
        date_str = row['match_date'].strftime('%Y-%m-%d')
        out.append(f">> {date_str} vs {row['opponent'].upper()} | {row['outcome']} ({row['score']})")
        sg = team_df[(team_df['match_date'] == row['match_date']) & (team_df['opponent'] == row['opponent'])].sort_values(by='game_id')
        for i, game in enumerate(sg.itertuples(), 1):
            ef = f"{game.early_frags:.0f}" if pd.notna(game.early_frags) else "N/A"
            fb = "Yes" if game.first_blood == 1 else "No "
            out.append(f"   G{i} | {game.result:<4} | Time: {game.duration_mins:>4.1f}m | Kills: {game.frags:>2.0f} | <10m Kills: {ef:>2} | 1st Blood: {fb}")
        out.append("")
            
    # --- HEAD TO HEAD STATS ---
    if h2h_opponent != "None":
        h2h_s = series_df[series_df['opponent'] == h2h_opponent]
        if h2h_s.empty:
            out.append(f"\nNo historical series found vs {h2h_opponent}.")
        else:
            h2h_g = team_df[team_df['opponent'] == h2h_opponent]
            h2h_e_str = f"{h2h_g['early_frags'].mean():.1f}" if not h2h_g['early_frags'].isna().all() else "N/A"
            out.append(f"=== HEAD-TO-HEAD vs {h2h_opponent.upper()} ===")
            out.append(f"Series Record:     {len(h2h_s[h2h_s['outcome']=='WIN'])}W - {len(h2h_s[h2h_s['outcome']=='LOSS'])}L")
            out.append(f"Game Record:       {h2h_g['win_val'].sum()}W - {len(h2h_g)-h2h_g['win_val'].sum()}L")
            out.append(f"Avg Total Frags:   {h2h_g['frags'].mean():.1f} per game")
            out.append(f"Avg Frags (<10m):  {h2h_e_str}")
            out.append(f"First Blood Rate:  {h2h_g['first_blood'].mean()*100:.0f}%\n")
            
            out.append("=== RECENT 3 H2H SERIES ===")
            for _, row in h2h_s.head(3).iterrows():
                date_str = row['match_date'].strftime('%Y-%m-%d')
                out.append(f">> {date_str} vs {row['opponent'].upper()} | {row['outcome']} ({row['score']})")
                h2h_sg = team_df[(team_df['match_date'] == row['match_date']) & (team_df['opponent'] == row['opponent'])].sort_values(by='game_id')
                for i, game in enumerate(h2h_sg.itertuples(), 1):
                    ef = f"{game.early_frags:.0f}" if pd.notna(game.early_frags) else "N/A"
                    fb = "Yes" if game.first_blood == 1 else "No "
                    out.append(f"   G{i} | {game.result:<4} | Time: {game.duration_mins:>4.1f}m | Kills: {game.frags:>2.0f} | <10m Kills: {ef:>2} | 1st Blood: {fb}")
                out.append("")
                
    # 5. Render Output
    st.code('\n'.join(out), language='text')