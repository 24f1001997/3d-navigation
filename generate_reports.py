import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def generate_report(logdir, stage_num):
    print(f"Generating reports for stage {stage_num} in {logdir}...")
    
    # Create output directory
    output_dir = os.path.join(logdir, 'graphs_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all train log files
    logfiles = glob.glob(os.path.join(logdir, f'_train_stage{stage_num}_*.txt')) + \
               glob.glob(os.path.join(logdir, 'backup_logs', f'_train_stage{stage_num}_*.txt'))
               
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

    # Combine, sort by episode, and drop duplicates
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip() # Strip whitespace from column names just in case
    df = df.sort_values(by='episode')
    df = df.drop_duplicates(subset=['episode'], keep='last')
    
    print(f"Loaded data up to episode {df['episode'].max()}")
    
    # Save the consolidated raw data to CSV (as requested)
    csv_path = os.path.join(output_dir, f'training_data_stage{stage_num}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved consolidated CSV data to: {csv_path}")

    # Add Boolean columns for outcome types
    # 1 = Success, 2 = Collision Wall, 3 = Collision Obstacle, 4 = Timeout
    df['is_success'] = (df['success'] == 1).astype(float)
    df['is_collision'] = ((df['success'] == 2) | (df['success'] == 3)).astype(float)
    df['is_timeout'] = (df['success'] == 4).astype(float)
    
    # We will generate graphs for these metrics
    metrics = {
        'Actor loss': 'avg_actor_loss',
        'Episode duration': 'duration',
        'Steps per episode': 'steps',
        'Success rate': 'is_success',
        'Collision rate': 'is_collision',
        'Timeout rate': 'is_timeout'
    }
    
    windows = [50, 100]
    
    for metric_title, col_name in metrics.items():
        if col_name not in df.columns:
            print(f"Warning: Column {col_name} not found, skipping {metric_title} graph.")
            continue
            
        for window in windows:
            # Calculate rolling average
            rolling_avg = df[col_name].rolling(window=window, min_periods=1).mean()
            
            plt.figure(figsize=(10, 6))
            # Plot the raw data lightly in the background (optional, but looks good)
            plt.plot(df['episode'], df[col_name], alpha=0.2, color='gray', label='Raw Data')
            # Plot the rolling average
            plt.plot(df['episode'], rolling_avg, color='blue', linewidth=2, label=f'{window}-Episode Average')
            
            plt.title(f'{metric_title} (average {window} episodes)')
            plt.xlabel('Episode')
            
            # Format Y axis properly based on metric type
            if 'rate' in metric_title:
                plt.ylabel('Rate (0.0 to 1.0)')
                plt.ylim([-0.05, 1.05])
            else:
                plt.ylabel(metric_title)
                
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save as PDF
            filename = f"{metric_title.replace(' ', '_')}_(average_{window}_episodes).pdf"
            pdf_path = os.path.join(output_dir, filename)
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            plt.close()
            
    print(f"All graphs have been saved to: {output_dir}")

if __name__ == '__main__':
    # You can change the directory and stage number here as needed
    LOG_DIR = '/home/pranav/turtlebot3_drlnav/src/turtlebot3_drl/model/GPREDDY/sac_0_stage_1/'
    STAGE = 1
    
    generate_report(LOG_DIR, STAGE)
