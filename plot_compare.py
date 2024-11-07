from __future__ import division
import random
import math
import pybullet as p
import sim_update
import numpy as np
import matplotlib.pyplot as plt
from main_rrt import rrt, run as run_rrt
from rrt_apf import rrt as rrt_apf, run as run_rrt_apf
from bi_rrt import bidirectional_rrt, run_bidirectional_rrt
from bi_rrt_apf import bidirectional_rrt as bi_rrt_apf, run_bidirectional_rrt as run_bi_rrt_apf
import time
from main_rrt import get_grasp_position_angle as get_gr
import os


class AlgorithmComparison:
    def __init__(self, num_trials=5):
        self.num_trials = num_trials
        self.object_shapes = ["assets/objects/rod.urdf"]
        self.env = sim_update.PyBulletSim(object_shapes=self.object_shapes)
        self.results = {
            'RRT': {'success_rate': 0, 'path_lengths': [], 'execution_times': []},
            'RRT-APF': {'success_rate': 0, 'path_lengths': [], 'execution_times': []},
            'Bi-RRT': {'success_rate': 0, 'path_lengths': [], 'execution_times': []},
            'Bi-RRT-APF': {'success_rate': 0, 'path_lengths': [], 'execution_times': []}
        }

    def run_comparison(self):
        # Tạo thư mục để lưu quỹ đạo
        if not os.path.exists('trajectories5'):
            os.makedirs('trajectories5')

        algorithms = {
            'RRT': (run_rrt, lambda env, start, goal, iters, delta, p: rrt(start, goal, iters, delta, p, env)),
            'RRT-APF': (run_rrt_apf, lambda env, start, goal, iters, delta, p: rrt_apf(start, goal, iters, delta, p, env)),
            'Bi-RRT': (run_bidirectional_rrt, lambda env, start, goal, iters, delta, p: bidirectional_rrt(env, start, goal, iters, delta, p)),
            'Bi-RRT-APF': (run_bi_rrt_apf, lambda env, start, goal, iters, delta, p: bi_rrt_apf(env, start, goal, iters, delta, p))
        }

        # Thêm success_indicators vào results
        for algo in self.results:
            self.results[algo]['success_indicators'] = np.zeros(self.num_trials)

        for algo_name, (run_func, algo_func) in algorithms.items():
            print(f"\nTesting {algo_name}...")
            for trial in range(self.num_trials):
                print(f"Trial {trial + 1}/{self.num_trials}")
                
                # Reset environment
                self.env.reset_objects()
                self.env.load_gripper()
                
                # Execute grasp
                object_id = self.env._objects_body_ids[0]
                position, grasp_angle = get_gr(object_id)
                grasp_success = self.env.execute_grasp(position, grasp_angle)
                
                if grasp_success:
                    # Measure execution time
                    start_time = time.time()
                    
                    # Run algorithm
                    path_conf = algo_func(
                        self.env,
                        self.env.robot_home_joint_config,
                        self.env.robot_goal_joint_config,
                        1000,  # MAX_ITERS
                        0.1,    # delta_q
                        0.5     # steer_goal_p
                    )
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Record results
                    if path_conf is not None:
                        # Lưu quỹ đạo
                        self.save_trajectory(path_conf, algo_name, trial)
                        
                        path_length = self.calculate_path_length(path_conf)
                        self.results[algo_name]['path_lengths'].append(path_length)
                        self.results[algo_name]['execution_times'].append(execution_time)
                        self.results[algo_name]['success_rate'] += 1
                        self.results[algo_name]['success_indicators'][trial] = 1

        # Convert success rates to percentages
        for algo in self.results:
            self.results[algo]['success_rate'] = (self.results[algo]['success_rate'] / self.num_trials) * 100

    def calculate_path_length(self, path):
        length = 0
        for i in range(1, len(path)):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        return length

    def save_trajectory(self, path_conf, algo_name, trial_num):
        """Lưu quỹ đạo vào file"""
        if path_conf is not None:
            filename = f"trajectories/{algo_name}_trial_{trial_num}.npy"
            np.save(filename, np.array(path_conf))
            return True
        return False

    def plot_results(self):
        # Create figure with subplots
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 12))

        algorithms = list(self.results.keys())
        colors = ['b', 'g', 'r', 'c']  # Màu cho mỗi thuật toán

        # 1. Success Rate over trials
        for i, algo in enumerate(algorithms):
            # Tính tỷ lệ thành công tích lũy
            cumulative_success = np.cumsum(self.results[algo]['success_indicators']) / \
                               np.arange(1, self.num_trials + 1)
            ax1.plot(range(1, self.num_trials + 1), cumulative_success * 100, 
                    label=algo, color=colors[i])
        ax1.set_title('Success Rate over Trials')
        ax1.set_xlabel('Number of Trials')
        ax1.set_ylabel('Success Rate (%)')
        ax1.legend()
        ax1.grid(True)

        # 2. Path Length over successful trials
        for i, algo in enumerate(algorithms):
            lengths = self.results[algo]['path_lengths']
            if lengths:
                trials = range(1, len(lengths) + 1)
                ax2.plot(trials, lengths, label=algo, color=colors[i])
        ax2.set_title('Path Length over Successful Trials')
        ax2.set_xlabel('Successful Trial Number')
        ax2.set_ylabel('Path Length')
        ax2.legend()
        ax2.grid(True)

        # 3. Execution Time over trials
        for i, algo in enumerate(algorithms):
            times = self.results[algo]['execution_times']
            if times:
                trials = range(1, len(times) + 1)
                ax3.plot(trials, times, label=algo, color=colors[i])
        ax3.set_title('Execution Time over Trials')
        ax3.set_xlabel('Trial Number')
        ax3.set_ylabel('Time (seconds)')
        ax3.legend()
        ax3.grid(True)

        # 4. Cumulative Average Path Length
        for i, algo in enumerate(algorithms):
            lengths = self.results[algo]['path_lengths']
            if lengths:
                cumulative_avg = np.cumsum(lengths) / np.arange(1, len(lengths) + 1)
                ax4.plot(range(1, len(lengths) + 1), cumulative_avg, 
                        label=algo, color=colors[i])
        ax4.set_title('Cumulative Average Path Length')
        ax4.set_xlabel('Number of Successful Trials')
        ax4.set_ylabel('Average Path Length')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('algorithm_comparison1.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_statistics(self):
        print("\nAlgorithm Comparison Statistics:")
        print("================================")
        
        for algo in self.results:
            print(f"\n{algo}:")
            print(f"Success Rate: {self.results[algo]['success_rate']:.2f}%")
            
            if self.results[algo]['path_lengths']:
                avg_length = np.mean(self.results[algo]['path_lengths'])
                std_length = np.std(self.results[algo]['path_lengths'])
                print(f"Average Path Length: {avg_length:.2f} ± {std_length:.2f}")
            
            if self.results[algo]['execution_times']:
                avg_time = np.mean(self.results[algo]['execution_times'])
                std_time = np.std(self.results[algo]['execution_times'])
                print(f"Average Execution Time: {avg_time:.2f}s ± {std_time:.2f}s")

def main():
    random.seed(42)  # For reproducibility
    comparison = AlgorithmComparison(num_trials=100)
    comparison.run_comparison()
    comparison.print_statistics()
    comparison.plot_results()

if __name__ == "__main__":
    main()