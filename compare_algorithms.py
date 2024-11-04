from __future__ import division
import random
import math
import pybullet as p
import sim_update
import numpy as np
import matplotlib.pyplot as plt
from main_rrt import rrt
from rrt_apf import rrt as rrt_apf
from bi_rrt import bidirectional_rrt
from bi_rrt_apf import bidirectional_rrt as bi_rrt_apf
import time

class AlgorithmComparison:
    def __init__(self, num_trials=50):
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
        MAX_ITERS = 10000
        delta_q = 0.1
        steer_goal_p = 0.5
        max_connection_distance = 0.3

        algorithms = {
            'RRT': rrt,
            'RRT-APF': rrt_apf,
            'Bi-RRT': bidirectional_rrt,
            'Bi-RRT-APF': bi_rrt_apf
        }

        for algo_name, algo_func in algorithms.items():
            print(f"\nTesting {algo_name}...")
            for trial in range(self.num_trials):
                print(f"Trial {trial + 1}/{self.num_trials}")
                
                # Reset environment
                self.env.reset_objects()
                self.env.load_gripper()
                
                try:
                    # Measure execution time
                    start_time = time.time()
                    
                    if algo_name in ['Bi-RRT', 'Bi-RRT-APF']:
                        # Bidirectional RRT variants
                        path_conf = algo_func(
                            self.env,
                            self.env.robot_home_joint_config,
                            self.env.robot_goal_joint_config,
                            MAX_ITERS,
                            delta_q,
                            steer_goal_p,
                            max_connection_distance
                        )
                    else:
                        # Regular RRT variants
                        path_conf = algo_func(
                            self.env.robot_home_joint_config,
                            self.env.robot_goal_joint_config,
                            MAX_ITERS,
                            delta_q,
                            steer_goal_p,
                            self.env
                        )
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Record results
                    if path_conf is not None:
                        path_length = self.calculate_path_length(path_conf)
                        self.results[algo_name]['path_lengths'].append(path_length)
                        self.results[algo_name]['execution_times'].append(execution_time)
                        self.results[algo_name]['success_rate'] += 1
                
                except Exception as e:
                    print(f"Error in {algo_name}, trial {trial}: {str(e)}")
                    continue

        # Convert success rates to percentages
        for algo in self.results:
            self.results[algo]['success_rate'] = (self.results[algo]['success_rate'] / self.num_trials) * 100

    def calculate_path_length(self, path):
        if not path:
            return 0
        length = 0
        for i in range(1, len(path)):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        return length

    def plot_results(self):
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Success Rate Comparison
        algorithms = list(self.results.keys())
        success_rates = [self.results[algo]['success_rate'] for algo in algorithms]
        ax1.bar(algorithms, success_rates)
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylabel('Success Rate (%)')
        ax1.tick_params(axis='x', rotation=45)

        # Path Length Comparison (Box Plot)
        path_lengths = [self.results[algo]['path_lengths'] for algo in algorithms]
        ax2.boxplot(path_lengths, labels=algorithms)
        ax2.set_title('Path Length Distribution')
        ax2.set_ylabel('Path Length')
        ax2.tick_params(axis='x', rotation=45)

        # Execution Time Comparison (Box Plot)
        exec_times = [self.results[algo]['execution_times'] for algo in algorithms]
        ax3.boxplot(exec_times, labels=algorithms)
        ax3.set_title('Execution Time Distribution')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('algorithm_comparison.png')
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
    comparison = AlgorithmComparison(num_trials=10)  # Reduced number of trials for testing
    comparison.run_comparison()
    comparison.print_statistics()
    comparison.plot_results()

if __name__ == "__main__":
    main() 