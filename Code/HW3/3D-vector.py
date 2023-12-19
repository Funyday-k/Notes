import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成数据
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)

# 计算矢量场
dx, dy = np.gradient(z) 
u = -dy
v = dx

fig = plt.figure()
ax = plt.axes(projection='3d')

# 绘制矢量场
ax.quiver(x, y, z, u, v, z, length=0.1, normalize=True)


ax.set_title('Vector Field')
plt.show()