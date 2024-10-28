from datetime import datetime, timedelta
import pandas as pd

def convert_jst_to_utc(jst_time_str):
    jst_offset = timedelta(hours=9)
    jst_time = datetime.strptime(jst_time_str, '%Y-%m-%d %H:%M:%S')
    utc_time = jst_time - jst_offset
    return utc_time

def load_meta_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp'], errors='coerce'), unit='s')
    return df

def get_session_data(session_df, session_id):
    selected_row = session_df.loc[session_df['session_id'] == session_id]
    location = selected_row['location'].values[0]
    start_time_jst = selected_row['start_time_JST'].values[0]
    end_time_jst = selected_row['end_time_JST'].values[0]
    return location, start_time_jst, end_time_jst