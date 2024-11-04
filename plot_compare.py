from __future__ import division
import sim_update
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from rrt_apf import rrt as rrt_with_apf
from main_rrt import rrt as rrt_without_apf  # Giả sử bạn có file rrt.py chứa RRT thuần túy
import json
import os

class PlannerComparison:
    def __init__(self, env):
        self.env = env
        self.results_dir = "comparison_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def measure_planning_time(self, q_init, q_goal, use_apf=False):
        """Đo thời gian thực thi của thuật toán"""
        start_time = time.time()
        if use_apf:
            path = rrt_with_apf(q_init, q_goal, 10000, 0.1, 0.5, self.env)
        else:
            path = rrt_without_apf(q_init, q_goal, 10000, 0.1, 0.5, self.env)
        end_time = time.time()
        return end_time - start_time, path

    def calculate_path_metrics(self, path):
        """Tính toán các chỉ số của đường đi"""
        if path is None:
            return None, None, None

        # Tổng độ dài đường đi
        total_length = 0
        # Độ mượt (tính bằng góc giữa các đoạn liên tiếp)
        smoothness = 0
        
        for i in range(1, len(path)):
            total_length += self.get_euclidean_distance(path[i-1], path[i])
            if i > 1:
                # Tính góc giữa các đoạn
                v1 = np.array(path[i-1]) - np.array(path[i-2])
                v2 = np.array(path[i]) - np.array(path[i-1])
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                smoothness += angle

        return total_length, len(path), smoothness

    def get_euclidean_distance(self, q1, q2):
        """Tính khoảng cách Euclidean giữa hai cấu hình"""
        return np.linalg.norm(np.array(q1) - np.array(q2))

    def run_comparison(self, num_trials=100):
        """Chạy so sánh giữa hai phương pháp"""
        results = {
            'rrt': {'success': 0, 'times': [], 'lengths': [], 'waypoints': [], 'smoothness': []},
            'rrt_apf': {'success': 0, 'times': [], 'lengths': [], 'waypoints': [], 'smoothness': []}
        }

        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}")

            # Test RRT without APF
            time_rrt, path_rrt = self.measure_planning_time(
                self.env.robot_home_joint_config,
                self.env.robot_goal_joint_config,
                use_apf=False
            )
            if path_rrt is not None:
                results['rrt']['success'] += 1
                results['rrt']['times'].append(time_rrt)
                length, waypoints, smoothness = self.calculate_path_metrics(path_rrt)
                results['rrt']['lengths'].append(length)
                results['rrt']['waypoints'].append(waypoints)
                results['rrt']['smoothness'].append(smoothness)

            # Test RRT with APF
            time_rrt_apf, path_rrt_apf = self.measure_planning_time(
                self.env.robot_home_joint_config,
                self.env.robot_goal_joint_config,
                use_apf=True
            )
            if path_rrt_apf is not None:
                results['rrt_apf']['success'] += 1
                results['rrt_apf']['times'].append(time_rrt_apf)
                length, waypoints, smoothness = self.calculate_path_metrics(path_rrt_apf)
                results['rrt_apf']['lengths'].append(length)
                results['rrt_apf']['waypoints'].append(waypoints)
                results['rrt_apf']['smoothness'].append(smoothness)

            self.env.reset_objects()

        # Lưu kết quả
        self.save_results(results, num_trials)
        return results

    def save_results(self, results, num_trials):
        """Lưu kết quả vào file"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{self.results_dir}/comparison_results_{timestamp}.json"
        
        # Chuyển đổi numpy arrays thành lists để có thể serialize
        serializable_results = {
            method: {
                metric: (list(values) if isinstance(values, (list, np.ndarray)) else values)
                for metric, values in data.items()
            }
            for method, data in results.items()
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f)

    def analyze_results(self, results, num_trials):
        """Phân tích và in kết quả"""
        for method in ['rrt', 'rrt_apf']:
            success_rate = results[method]['success'] / num_trials * 100
            avg_time = np.mean(results[method]['times']) if results[method]['times'] else 0
            avg_length = np.mean(results[method]['lengths']) if results[method]['lengths'] else 0
            avg_waypoints = np.mean(results[method]['waypoints']) if results[method]['waypoints'] else 0
            avg_smoothness = np.mean(results[method]['smoothness']) if results[method]['smoothness'] else 0

            print(f"\nResults for {method.upper()}:")
            print(f"Success Rate: {success_rate:.2f}%")
            print(f"Average Planning Time: {avg_time:.3f} seconds")
            print(f"Average Path Length: {avg_length:.3f}")
            print(f"Average Number of Waypoints: {avg_waypoints:.1f}")
            print(f"Average Smoothness: {avg_smoothness:.3f}")

    def plot_comparison(self, results):
        """Vẽ đồ thị so sánh"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Plot planning times
        self._create_boxplot(axes[0,0], 
                           [results['rrt']['times'], results['rrt_apf']['times']],
                           ['RRT', 'RRT+APF'],
                           'Planning Time Distribution',
                           'Time (seconds)')

        # Plot path lengths
        self._create_boxplot(axes[0,1],
                           [results['rrt']['lengths'], results['rrt_apf']['lengths']],
                           ['RRT', 'RRT+APF'],
                           'Path Length Distribution',
                           'Path Length')

        # Plot number of waypoints
        self._create_boxplot(axes[1,0],
                           [results['rrt']['waypoints'], results['rrt_apf']['waypoints']],
                           ['RRT', 'RRT+APF'],
                           'Number of Waypoints Distribution',
                           'Number of Waypoints')

        # Plot smoothness
        self._create_boxplot(axes[1,1],
                           [results['rrt']['smoothness'], results['rrt_apf']['smoothness']],
                           ['RRT', 'RRT+APF'],
                           'Path Smoothness Distribution',
                           'Smoothness (radians)')

        # Plot success rates
        success_rates = [
            results['rrt']['success'] / len(results['rrt']['times']) * 100,
            results['rrt_apf']['success'] / len(results['rrt_apf']['times']) * 100
        ]
        axes[2,0].bar(['RRT', 'RRT+APF'], success_rates)
        axes[2,0].set_title('Success Rate')
        axes[2,0].set_ylabel('Success Rate (%)')

        # Hide the last unused subplot
        axes[2,1].set_visible(False)

        plt.tight_layout()
        
        # Lưu plot
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"{self.results_dir}/comparison_plot_{timestamp}.png")
        plt.show()

    def _create_boxplot(self, ax, data, labels, title, ylabel):
        """Helper function để tạo boxplot"""
        ax.boxplot(data, labels=labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

def main():
    # Khởi tạo môi trường
    object_shapes = ["assets/objects/rod.urdf"]
    env = sim_update.PyBulletSim(object_shapes=object_shapes)
    
    # Khởi tạo và chạy so sánh
    comparison = PlannerComparison(env)
    num_trials = 20  # Có thể điều chỉnh số lượng thử nghiệm
    results = comparison.run_comparison(num_trials)
    
    # Phân tích và hiển thị kết quả
    comparison.analyze_results(results, num_trials)
    comparison.plot_comparison(results)

if __name__ == "__main__":
    main()