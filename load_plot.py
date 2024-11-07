import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_from_saved_trajectories():
    # Tạo figure với subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Định nghĩa các thuật toán và màu sắc
    algorithms = ['RRT', 'RRT-APF', 'Bi-RRT', 'Bi-RRT-APF']
    colors = ['b', 'g', 'r', 'c']
    
    # Dictionary để lưu kết quả
    results = {algo: {'paths': [], 'lengths': []} for algo in algorithms}
    
    # Load tất cả các file trajectory
    for algo in algorithms:
        # Lấy tất cả các file của thuật toán
        files = glob.glob(f'trajectories/{algo}_trial_*.npy')
        
        # Load và tính toán độ dài cho mỗi quỹ đạo
        for file in sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
            try:
                trajectory = np.load(file)
                results[algo]['paths'].append(trajectory)
                
                # Tính độ dài quỹ đạo
                length = 0
                for i in range(1, len(trajectory)):
                    length += np.linalg.norm(trajectory[i] - trajectory[i-1])
                results[algo]['lengths'].append(length)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    # 1. Success Rate over trials
    # for i, algo in enumerate(algorithms):
    #     num_successful = len(results[algo]['paths'])
    #     success_rate = [j/50*100 for j in range(1, num_successful + 1)]  # Giả sử tổng số trials là 50
    #     ax1.plot(range(1, num_successful + 1), success_rate, 
    #             label=algo, color=colors[i])
    # ax1.set_title('Success Rate over Trials')
    # ax1.set_xlabel('Number of Trials')
    # ax1.set_ylabel('Success Rate (%)')
    # ax1.legend()
    # ax1.grid(True)

    # 2. Path Length Distribution
    for i, algo in enumerate(algorithms):
        lengths = results[algo]['lengths']
        if lengths:
            ax2.plot(range(1, len(lengths) + 1), lengths, 
                    label=algo, color=colors[i])
    ax2.set_title('Path Length Distribution')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Path Length')
    ax2.legend()
    ax2.grid(True)

    # 3. Average Path Length per Algorithm
    avg_lengths = [np.mean(results[algo]['lengths']) if results[algo]['lengths'] else 0 
                  for algo in algorithms]
    std_lengths = [np.std(results[algo]['lengths']) if results[algo]['lengths'] else 0 
                  for algo in algorithms]
    ax3.bar(algorithms, avg_lengths, yerr=std_lengths, color=colors)
    ax3.set_title('Average Path Length per Algorithm')
    ax3.set_ylabel('Average Length')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True)

    # 4. Cumulative Average Path Length
    for i, algo in enumerate(algorithms):
        lengths = results[algo]['lengths']
        if lengths:
            cumulative_avg = np.cumsum(lengths) / np.arange(1, len(lengths) + 1)
            ax4.plot(range(1, len(lengths) + 1), cumulative_avg, 
                    label=algo, color=colors[i])
    ax4.set_title('Cumulative Average Path Length')
    ax4.set_xlabel('Number of Trials')
    ax4.set_ylabel('Average Path Length')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # In thống kê
    print("\nStatistics from saved trajectories:")
    print("===================================")
    for algo in algorithms:
        lengths = results[algo]['lengths']
        if lengths:
            print(f"\n{algo}:")
            print(f"Number of successful trials: {len(lengths)}")
            print(f"Average path length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
            print(f"Min path length: {np.min(lengths):.2f}")
            print(f"Max path length: {np.max(lengths):.2f}")

if __name__ == "__main__":
    plot_from_saved_trajectories()