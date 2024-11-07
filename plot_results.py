import matplotlib.pyplot as plt
import numpy as np

def plot_path_comparison(rrt_lengths, rrt_apf_lengths, birrt_lengths, birrt_apf_lengths):
    # Xử lý dữ liệu
    algorithms = ['RRT', 'RRT-APF', 'Bi-RRT', 'Bi-RRT-APF']
    all_lengths = [rrt_lengths, rrt_apf_lengths, birrt_lengths, birrt_apf_lengths]
    
    # Tính toán các thông số cho mỗi thuật toán
    stats = []
    for lengths in all_lengths:
        # Lọc bỏ None values
        valid_lengths = [l for l in lengths if l is not None]
        if valid_lengths:
            stats.append({
                'mean': np.mean(valid_lengths),
                'std': np.std(valid_lengths),
                'success_rate': len(valid_lengths) / len(lengths) * 100,
                'min': np.min(valid_lengths),
                'max': np.max(valid_lengths)
            })
        else:
            stats.append({
                'mean': 0,
                'std': 0,
                'success_rate': 0,
                'min': 0,
                'max': 0
            })

    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Box plot cho độ dài đường đi
    bp_data = [[l for l in lengths if l is not None] for lengths in all_lengths]
    ax1.boxplot(bp_data, labels=algorithms)
    ax1.set_title('Độ dài đường đi')
    ax1.set_ylabel('Độ dài')
    ax1.grid(True)

    # 2. Bar plot cho tỷ lệ thành công
    success_rates = [stat['success_rate'] for stat in stats]
    ax2.bar(algorithms, success_rates)
    ax2.set_title('Tỷ lệ tìm được đường đi')
    ax2.set_ylabel('Tỷ lệ thành công (%)')
    ax2.grid(True)

    # In thông số chi tiết
    print("\nThống kê chi tiết:")
    for alg, stat in zip(algorithms, stats):
        print(f"\n{alg}:")
        print(f"Độ dài trung bình: {stat['mean']:.2f}")
        print(f"Độ lệch chuẩn: {stat['std']:.2f}")
        print(f"Tỷ lệ thành công: {stat['success_rate']:.1f}%")
        print(f"Độ dài min: {stat['min']:.2f}")
        print(f"Độ dài max: {stat['max']:.2f}")

    plt.tight_layout()
    plt.show()

# Sử dụng hàm
# Thay thế các giá trị này bằng kết quả thực tế từ các lần chạy của bạn
rrt_lengths = [...]  # Kết quả từ main_rrt.py
rrt_apf_lengths = [...]  # Kết quả từ rrt_apf.py
birrt_lengths = [...]  # Kết quả từ bi_rrt.py
birrt_apf_lengths = [...]  # Kết quả từ bi_rrt_apf.py

plot_path_comparison(rrt_lengths, rrt_apf_lengths, birrt_lengths, birrt_apf_lengths)