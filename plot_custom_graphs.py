import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Use non-interactive backend for generating plots
matplotlib.use('Agg')

def create_custom_graphs(logdir, stage_num, ep_start, ep_end):
    # 1. Create output directory with stage and episode range
    output_dir = os.path.join(logdir, f'custom_graphs_stage{stage_num}_ep{ep_start}_to_{ep_end}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Find all train log files
    logfiles = glob.glob(os.path.join(logdir, '_train_stage*.txt')) + \
               glob.glob(os.path.join(logdir, 'backup_logs', '_train_stage*.txt'))
               
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

    # 3. Combine, sort, drop duplicates, and FILTER by episode range
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.sort_values(by='episode')
    df = df.drop_duplicates(subset=['episode'], keep='last')
    
    df = df[(df['episode'] >= ep_start) & (df['episode'] <= ep_end)]
    
    if df.empty:
        print(f"No data found for episodes {ep_start} to {ep_end}.")
        return
        
    print(f"Found {len(df)} episodes in range {ep_start} - {ep_end}.")

    episodes = df['episode'].values

    # ---------------------------------------------------------
    # FEATURE A: The 2x2 _figure.png style grid
    # ---------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(18.5, 10.5))
    titles = ['Outcomes (Cumulative)', 'Avg Critic Loss', 'Avg Actor Loss', 'Avg Reward (10-ep Rolling)']
    for i in range(4):
        a = ax[int(i/2)][int(i%2!=0)]
        a.set_title(titles[i])
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # Top-Left: Outcomes (Cumulative)
    # 1=Success, 2=Wall, 3=Dynamic, 4=Timeout, 5=Tumble
    labels = {1: 'Success', 2: 'Collision Wall', 3: 'Collision Dynamic', 4: 'Timeout', 5: 'Tumble'}
    colors = {1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y'}
    
    # Calculate cumulative counts for the specific range
    cum_counts = {1: [], 2: [], 3: [], 4: [], 5: []}
    running = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for outcome in df['success']:
        val = int(outcome)
        if val in running:
            running[val] += 1
        for k in running.keys():
            cum_counts[k].append(running[k])
            
    for k in labels.keys():
        ax[0][0].plot(episodes, cum_counts[k], color=colors[k], label=labels[k])
    ax[0][0].legend()
    
    # Top-Right: Critic Loss
    ax[0][1].plot(episodes, df['avg_critic_loss'].values, color='gray')
    
    # Bottom-Left: Actor Loss
    ax[1][0].plot(episodes, df['avg_actor_loss'].values, color='gray')
    
    # Bottom-Right: Reward over 10 episodes (rolling average)
    reward_rolling = df['reward'].rolling(window=10, min_periods=1).mean()
    ax[1][1].plot(episodes, reward_rolling.values, color='gray')
    
    grid_path = os.path.join(output_dir, '_figure_subset.png')
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close()
    print(f"Saved 2x2 grid plot to {grid_path}")

    # ---------------------------------------------------------
    # FEATURE B: Individual PDFs for requested metrics
    # ---------------------------------------------------------
    
    # Add Boolean columns for rates
    df['is_success'] = (df['success'] == 1).astype(float)
    df['is_collision'] = ((df['success'] == 2) | (df['success'] == 3)).astype(float)
    df['is_timeout'] = (df['success'] == 4).astype(float)
    
    # Add cumulative columns for "number of" plots
    df['cum_successes'] = cum_counts[1]
    df['cum_collisions'] = np.array(cum_counts[2]) + np.array(cum_counts[3])
    
    metrics_to_plot = [
        ('Actor Loss', 'avg_actor_loss', 'raw'),
        ('Critic Loss', 'avg_critic_loss', 'raw'),
        ('Episode Duration', 'duration', 'rolling_avg'),
        ('Steps Per Episode', 'steps', 'rolling_avg'),
        ('Number of Successes (Cumulative)', 'cum_successes', 'raw'),
        ('Number of Collisions (Cumulative)', 'cum_collisions', 'raw'),
        ('Timeout Rate', 'is_timeout', 'rolling_avg'),
        ('Average Reward', 'reward', 'rolling_avg')
    ]
    
    windows = [50, 100]
    
    for title, col, plot_type in metrics_to_plot:
        if col not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'raw':
            plt.plot(episodes, df[col].values, color='blue', linewidth=2)
            plt.title(f'{title} (Episodes {ep_start}-{ep_end})')
            plt.xlabel('Episode')
            plt.grid(True, linestyle='--', alpha=0.7)
            filename = f"{title.replace(' ', '_')}.pdf"
            plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
            
        elif plot_type == 'rolling_avg':
            for w in windows:
                plt.figure(figsize=(10, 6))
                
                # Plot raw data lightly in background
                plt.plot(episodes, df[col].values, alpha=0.2, color='gray', label='Raw Data')
                
                # Plot rolling average
                rolling = df[col].rolling(window=w, min_periods=1).mean()
                plt.plot(episodes, rolling.values, color='blue', linewidth=2, label=f'{w}-Episode Avg')
                
                plt.title(f'{title} (Average {w} Episodes) (Episodes {ep_start}-{ep_end})')
                plt.xlabel('Episode')
                if 'Rate' in title:
                    plt.ylim([-0.05, 1.05])
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                filename = f"{title.replace(' ', '_')}_(average_{w}_episodes).pdf"
                plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
                plt.close()
        else:
            plt.close()

    # ---------------------------------------------------------
    # FEATURE C: Summary CSV
    # ---------------------------------------------------------
    total_episodes = len(df)
    total_successes = int(df['is_success'].sum())
    total_collisions = int(df['is_collision'].sum())
    total_timeouts = int(df['is_timeout'].sum())
    
    summary_data = {
        'Total Episodes': [total_episodes],
        'Total Successes': [total_successes],
        'Total Collisions': [total_collisions],
        'Total Timeouts': [total_timeouts]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary statistics CSV to {summary_csv_path}")

    print(f"All individual PDFs generated in {output_dir}")


if __name__ == '__main__':
    # ==========================================
    # CONFIGURATION VARIABLES
    # ==========================================
    LOG_DIR = '/home/pranav/turtlebot3_drlnav/src/turtlebot3_drl/model/GPREDDY/sac_6_stage_1/'
    
    # 1. Episode start and end values
    EPISODE_START = 1
    EPISODE_END = 500
    
    # 2. Stage variable
    STAGE_NUM = 1
    
    create_custom_graphs(LOG_DIR, STAGE_NUM, EPISODE_START, EPISODE_END)
