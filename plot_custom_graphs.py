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
    grid_titles = [
        f'outcomes per episode from episode {ep_start} to episode {ep_end}',
        f'critic loss per episode from episode {ep_start} to episode {ep_end}',
        f'actor loss per episode from episode {ep_start} to episode {ep_end}',
        f'average reward per episode (10-episode average) from episode {ep_start} to episode {ep_end}'
    ]
    grid_ylabels = [
        'Outcomes',
        'Critic loss',
        'Actor loss',
        'Average reward'
    ]
    for i in range(4):
        a = ax[int(i/2)][int(i%2!=0)]
        a.set_title(grid_titles[i], fontsize=12)
        a.set_xlabel('episode number', fontsize=11)
        a.set_ylabel(grid_ylabels[i], fontsize=11)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.grid(True, linestyle='--', alpha=0.7)
        
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
    
    plt.tight_layout()
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
        {
            'col': 'avg_actor_loss',
            'plot_type': 'raw',
            'ylabel': 'Actor loss',
            'title': f'actor loss per episode from episode {ep_start} to episode {ep_end}',
            'filename': 'Actor_Loss.pdf'
        },
        {
            'col': 'avg_critic_loss',
            'plot_type': 'raw',
            'ylabel': 'Critic loss',
            'title': f'critic loss per episode from episode {ep_start} to episode {ep_end}',
            'filename': 'Critic_Loss.pdf'
        },
        {
            'col': 'duration',
            'plot_type': 'raw',
            'ylabel': 'Episode duration',
            'title': f'episode duration per episode from episode {ep_start} to episode {ep_end}',
            'filename': 'Episode_Duration.pdf'
        },
        {
            'col': 'duration',
            'plot_type': 'rolling_avg',
            'ylabel': 'Average duration',
            'title_template': 'average duration per episode ({w}-episode average) from episode {ep_start} to episode {ep_end}',
            'filename_template': 'Average_Duration_(average_{w}_episodes).pdf'
        },
        {
            'col': 'steps',
            'plot_type': 'raw',
            'ylabel': 'Steps per episode',
            'title': f'steps per episode from episode {ep_start} to episode {ep_end}',
            'filename': 'Steps_Per_Episode.pdf'
        },
        {
            'col': 'steps',
            'plot_type': 'rolling_avg',
            'ylabel': 'Steps per episode',
            'title_template': 'steps per episode ({w}-episode average) from episode {ep_start} to episode {ep_end}',
            'filename_template': 'Steps_Per_Episode_(average_{w}_episodes).pdf'
        },
        {
            'col': 'reward',
            'plot_type': 'rolling_avg',
            'ylabel': 'Average reward',
            'title_template': 'average reward per episode ({w}-episode average) from episode {ep_start} to episode {ep_end}',
            'filename_template': 'Average_Reward_(average_{w}_episodes).pdf'
        },
        {
            'col': 'is_timeout',
            'plot_type': 'rolling_avg',
            'ylabel': 'Timeout rate',
            'title_template': 'timeout rate per episode ({w}-episode average) from episode {ep_start} to episode {ep_end}',
            'filename_template': 'Timeout_Rate_(average_{w}_episodes).pdf',
            'ylim': [-0.05, 1.05]
        },
        {
            'col': 'cum_successes',
            'plot_type': 'raw',
            'ylabel': 'Number of successes',
            'title': f'number of successes per episode from episode {ep_start} to episode {ep_end}',
            'filename': 'Number_of_Successes_(Cumulative).pdf'
        },
        {
            'col': 'cum_collisions',
            'plot_type': 'raw',
            'ylabel': 'Number of collisions',
            'title': f'number of collisions per episode from episode {ep_start} to episode {ep_end}',
            'filename': 'Number_of_Collisions_(Cumulative).pdf'
        }
    ]
    
    windows = [50, 100]
    
    for metric in metrics_to_plot:
        col = metric['col']
        if col not in df.columns:
            continue
            
        if metric['plot_type'] == 'raw':
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, df[col].values, color='blue', linewidth=2)
            plt.title(metric['title'], fontsize=13)
            plt.xlabel('episode number', fontsize=11)
            plt.ylabel(metric['ylabel'], fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, metric['filename']), format='pdf', bbox_inches='tight')
            plt.close()
            
        elif metric['plot_type'] == 'rolling_avg':
            for w in windows:
                plt.figure(figsize=(10, 6))
                
                # Plot raw data lightly in background
                plt.plot(episodes, df[col].values, alpha=0.2, color='gray', label='Raw Data')
                
                # Plot rolling average
                rolling = df[col].rolling(window=w, min_periods=1).mean()
                plt.plot(episodes, rolling.values, color='blue', linewidth=2, label=f'{w}-Episode Avg')
                
                title_str = metric['title_template'].format(w=w, ep_start=ep_start, ep_end=ep_end)
                plt.title(title_str, fontsize=13)
                plt.xlabel('episode number', fontsize=11)
                plt.ylabel(metric['ylabel'], fontsize=11)
                
                if 'ylim' in metric:
                    plt.ylim(metric['ylim'])
                    
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                filename = metric['filename_template'].format(w=w)
                plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
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
