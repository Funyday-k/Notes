# PDE

## 1 概论

### 1.1 线性方程

$L[c_1u_1+c_2u_2]=c_1L[u_1]+c_2L[u_2]$

### 1.2 三种数理方程
泊松方程

亥姆霍兹方程

$\nabla^2u+k^2u=0$

### 1.3 解法

1. 可通过函数变换&变量变换等技巧，把方程换成可直接求解的方程
2. 分离变量法
3. 积分变换（对时间变换——拉普拉斯；对空间变换——傅立叶变换）
4. $\delta$函数法：格林函数应用
5. 变分法（近似解法）

## 2 行波法

$$\frac{\partial ^{2}u}{\partial t^{2}}-a^{2}\frac{\partial^{2 u}}{\partial x^{2}}=0, -\infty<x<\infty, t>0\\
u(x,t)|_{t=0}=\phi(x),\frac{\partial u(x,t)}{\partial t}|_{t=0}
$$

达朗贝尔解

$$ u=f(x-at_)+g(x+at)\\
=\frac{1}{2}[\phi(x-at)+\phi(x+at)]+\frac{1}{2a}\int^{x+at}_{x-at}\psi (\xi)d\xi$$

1. 达朗贝尔公式
2. 反射波和延拓法
3. 强迫振动和冲凉定理法
4. 三维的波动法

## 3 分离变量法：有界弦自由振动

$$\frac{\partial^{2} u}{\partial x^{2}}-a^{2}\frac{\partial u}{\partial x^{2}}=0$$

分离变量 $u(x,t)=X(x)T(t)$

得到本征值问题

本征函数展开法

