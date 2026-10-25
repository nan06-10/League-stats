import streamlit as st
import sqlite3
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 1. Page Configuration for Mobile
st.set_page_config(page_title="LoL Esports Intel", page_icon="🎮", layout="centered")
st.title("🏆 LoL Pro Match Predictor")

# 2. Load and Cache Data (Runs only once to keep the app lightning fast)
@st.cache_data
def load_data():
    # Load SQLite Database
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

    # Load CSV Database
    try:
        df_csv_raw = pd.read_csv('2026_LoL_esports_match_data_from_OraclesElixir.csv', low_memory=False)
        df_csv_team = df_csv_raw[df_csv_raw['position'] == 'team']
        df_csv = pd.DataFrame({
            'match_date': pd.to_datetime(df_csv_team['date']).dt.strftime('%Y-%m-%d'),
            'game_id': df_csv_team['gameid'], 'league': df_csv_team['league'],
            'team_name': df_csv_team['teamname'], 'opponent': df_csv_team['opponent'],
            'result': df_csv_team['result'].apply(lambda x: 'Win' if str(x) in ['1', '1.0', 'True', 'TRUE'] else 'Loss'),
            'duration_mins': df_csv_team['gamelength'] / 60.0, 'frags': df_csv_team['kills'],
            'first_blood': df_csv_team['firstblood'].apply(lambda x: 1 if str(x) in ['1', '1.0', 'True', 'TRUE'] else 0),
            'early_frags': df_csv_team['killsat10'] if 'killsat10' in df_csv_team.columns else None
        })
    except FileNotFoundError:
        df_csv = pd.DataFrame()

    # Combine Data
    master_df = pd.concat([df_sql, df_csv]).drop_duplicates(subset=['game_id', 'team_name']).dropna(subset=['team_name'])
    
    # Active Major Teams Filter
    major_leagues = ['LPL', 'LCK', 'LEC', 'LCS', 'LTA', 'LCP', 'WCS', 'MSI']
    banned_leagues = ['LCKC', 'LIT', 'ROL', 'HLL', 'AL', 'NLC', 'Academy', 'Challengers', 'NACL', 'LDL', 'LFL', 'PRM', 'Cup']
    master_df = master_df[master_df['league'].str.contains('|'.join(major_leagues), case=False, na=False)]
    master_df = master_df[~master_df['league'].str.contains('|'.join(banned_leagues), case=False, na=False)]
    
    # Active Teams Filter (Played in the last 120 days)
    master_df['match_date'] = pd.to_datetime(master_df['match_date'], errors='coerce')
    latest_db_date = master_df['match_date'].max()
    team_last_match = master_df.groupby('team_name')['match_date'].max()
    active_teams = team_last_match[team_last_match >= (latest_db_date - pd.Timedelta(days=120))].index
    master_df = master_df[master_df['team_name'].isin(active_teams)]
    
    return master_df.sort_values(by='match_date', ascending=False)

# Load data into the app visually
with st.spinner('Loading Databases...'):
    master_df = load_data()
    available_teams = sorted(master_df['team_name'].unique())

# 3. User Interface (Dropdown Menus)
col1, col2 = st.columns(2)
with col1:
    primary_team = st.selectbox("Select Primary Team:", available_teams)
with col2:
    h2h_opponent = st.selectbox("Select H2H Opponent:", ["None"] + available_teams)

# 4. Search Logic
if st.button("Generate Intel Report", use_container_width=True):
    team_df = master_df[master_df['team_name'] == primary_team].copy()
    team_df['win_val'] = (team_df['result'] == 'Win').astype(int)
    
    # Group games into Series (BO3 / BO5)
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
    
    # Build Text Output Block
    out = []
    
    # --- OVERALL STATS ---
    series_wins = len(series_df[series_df['outcome'] == 'WIN'])
    series_losses = len(series_df[series_df['outcome'] == 'LOSS'])
    game_wins = team_df['win_val'].sum()
    game_losses = len(team_df) - game_wins
    e_frags_str = f"{team_df['early_frags'].mean():.1f}" if 'early_frags' in team_df.columns and not team_df['early_frags'].isna().all() else "N/A"
    
    out.append(f"=== OVERALL AVERAGES: {primary_team.upper()} ===")
    out.append(f"Series Record:     {series_wins}W - {series_losses}L")
    out.append(f"Game Record:       {game_wins}W - {game_losses}L")
    out.append(f"Avg Total Frags:   {team_df['frags'].mean():.1f} per game")
    out.append(f"Avg Frags (<10m):  {e_frags_str}")
    out.append(f"First Blood Rate:  {team_df['first_blood'].mean()*100:.0f}%\n")
    
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