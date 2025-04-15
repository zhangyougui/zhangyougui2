# 仇河东 324085503103 24机械1班 github：https://github.com/Key407/K
## 拒绝采样的方法模拟样本.py
```python
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
```

### 功能描述
1.参数定义：首先根据已知条件定义了公式中的参数 R_sun、A 、alpha 、b 、R_plt 。
2.目标函数定义：target_function 函数按照给定公式编写，用于计算给定 r 值处的函数值。
3.提议分布与 M 值确定：
  a.选择均匀分布作为提议分布，通过 proposal_distribution 函数从下限 lower_bound 到上限 upper_bound 之间均匀采样。
  b.在 [lower_bound, upper_bound] 区间内取 1000 个点（r_values ），计算目标函数在这些点的值，取最大值作为 M ，用于拒绝采样中的接受 - 拒绝判断。
4.拒绝采样过程：通过 while 循环不断从提议分布中生成候选样本 candidate ，同时生成一个 [0, 1] 之间的均匀随机数 u ，如果 u 小于目标函数在 candidate 处的值与 M 的比值，则接受该样本加入到 samples 列表中，直到样本数量达到 13 万。
5.绘图：
  a.使用 np.linspace 生成一系列 r 值（r_plot ），计算目标函数在这些值上的结果并绘制目标函数曲线。
  b.用 plt.hist 绘制样本的统计分布直方图，设置 bins = 100 表示直方图的区间数量，density = True 表示归一化直方图使其表示概率密度，最后添加坐标轴标签、标题和图例并展示图形。
### 使用方法
1、保存拒绝采样的方法模拟样本.py文件

2、在终端或命令行中执行启动程序

3、生成图片




