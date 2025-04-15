import numpy as np
import matplotlib.pyplot as plt

# 参数设定
R_sun = 8.3
A = 20.41
alpha = 9.03
b = 13.99
R_plt = 3.76

# 定义目标函数
def target_function(r):
    return (A / (R_sun + R_plt)) * ((r + R_plt) / (R_sun + R_plt)) ** alpha * np.exp(-b * ((r - R_sun) / (R_sun + R_plt)) ** 2)

# 确定采样区间和提议分布（这里用均匀分布作为提议分布）
lower_bound = 0
upper_bound = 20
proposal_distribution = lambda: np.random.uniform(lower_bound, upper_bound)

# 找到M值（这里简单通过在区间内取一些点估算最大值来确定M）
r_values = np.linspace(lower_bound, upper_bound, 1000)
max_f = np.max(target_function(r_values))
M = max_f  # 这里假设提议分布在区间内概率密度为1，实际取决于提议分布具体形式

# 拒绝采样
samples = []
while len(samples) < 130000:
    candidate = proposal_distribution()
    u = np.random.uniform(0, 1)
    if u < target_function(candidate) / M:
        samples.append(candidate)

samples = np.array(samples)

# 绘制目标函数曲线
r_plot = np.linspace(lower_bound, upper_bound, 1000)
plt.plot(r_plot, target_function(r_plot), label='Target Function')

# 绘制样本的统计分布直方图
plt.hist(samples, bins=100, density=True, label='Samples Histogram')
plt.xlabel('r')
plt.ylabel('Density')
plt.title('Samples from Target Function via Rejection Sampling')
plt.legend()
plt.show()