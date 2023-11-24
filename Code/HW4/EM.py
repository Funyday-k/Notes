import numpy as np 
from scipy import integrate
import matplotlib.pyplot as plt

a = 5 # 圆环半径
q = 1 #电荷量

def lambda_t(t):
    return q/(2*np.pi*a) 

def potential(x, y, z):
    k = 9 #库伦系数
    r = np.sqrt(x**2 + (y - a)**2 + z**2)
    return k * integrate.quad(lambda t:lambda_t(t)/r, 0, 2 * np.pi)

# # 求解电势梯度
# def dphi_dx(x, y, z):
#     delta = 0.001
#     return (potential(x+delta, y, z) - potential(x-delta, y, z))/(2*delta)

# def dphi_dy(x, y, z):
#     delta = 0.001
#     return (potential(x, y+delta, z) - potential(x, y-delta, z))/(2*delta)  

# def dphi_dz(x, y, z):
#     delta = 0.001 
#     return (potential(x, y, z+delta) - potential(x, y, z-delta))/(2*delta)

print(potential(5,0,0)[0])

# 生成坐标数据
x = np.linspace(-2*a, 2*a, 30)
y = np.linspace(-2*a, 2*a, 30)
X, Y = np.meshgrid(x, y)

# 电场分量    
# Ex = -dphi_dx(X, Y, 0) 
# Ey = -dphi_dy(X, Y, 0)
# Ez = -dphi_dz(X, Y, 0)


# # 计算电势和电场分布
V = potential(X, Y, 0)[0]

# # 绘制电势分布图
plt.contourf(X, Y, V)  
plt.colorbar()
plt.title('Electric Potential')

# plt.quiver(X, Y, Ex, Ey)
# plt.title('Electric Field')
plt.show()