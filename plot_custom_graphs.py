import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def main():
    logdir = '/home/pranav/turtlebot3_drlnav/src/turtlebot3_drl/model/GPREDDY/sac_0_stage_1/'
    
    # Find all train log files in the main directory and backup_logs directory
    logfiles = glob.glob(os.path.join(logdir, '_train_stage1_*.txt')) + \
               glob.glob(os.path.join(logdir, 'backup_logs', '_train_stage1_*.txt'))
               
    if not logfiles:
        print(f"Error: No log files found in {logdir}")
        return

    dfs = []
    for f in logfiles:
        try:
            df = pd.read_csv(f, skipinitialspace=True)
            if not df.empty and 'episode' in df.columns:
                dfs.append(df)
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")
            
    if not dfs:
        print("No valid data found in log files.")
        return

    # Combine, sort by episode, and drop duplicates (keep the last recorded data for any given episode)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by='episode')
    df = df.drop_duplicates(subset=['episode'], keep='last')
    
    print(f"Loaded data up to episode {df['episode'].max()}")

    # 1. Number of successes vs episodes (cumulative successes)
    df['is_success'] = (df['success'] == 1)
    df['cumulative_successes'] = df['is_success'].cumsum()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'].values, df['cumulative_successes'].values, label='Cumulative Successes')
    plt.title('Cumulative Successes vs Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Number of Successes')
    plt.grid(True)
    plt.savefig('successes_vs_episodes.png')
    plt.close()
    
    # 2. Episode duration vs episodes
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'].values, df['duration'].values, alpha=0.5, label='Duration')
    # Add a moving average for better visualization
    plt.plot(df['episode'].values, df['duration'].rolling(window=50, min_periods=1).mean().values, label='50-ep Moving Avg', color='red')
    plt.title('Episode Duration vs Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Duration (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('duration_vs_episodes.png')
    plt.close()
    
    # 3. Steps per episode vs episodes
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'].values, df['steps'].values, alpha=0.5, label='Steps')
    plt.plot(df['episode'].values, df['steps'].rolling(window=50, min_periods=1).mean().values, label='50-ep Moving Avg', color='red')
    plt.title('Steps per Episode vs Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('steps_vs_episodes.png')
    plt.close()

    print("Graphs generated successfully as PNG files.")

if __name__ == '__main__':
    main()
