# QM

## 常数

1. 普朗克常数 $\hbar$
2. 引力常数 $G$
3. 光速 $c$

## 介绍

普朗克公式
$$\rho(\nu,T)=\frac{8\pi h}{c^{3}}\frac{\nu^{3}}{e^{\frac{h\nu}{K_{B}T}}-1}$$

$h$ 为普朗克常数; 爱因斯坦关系：$E=\hbar \omega=h\nu$

### 原子稳定性

波尔模型：

1. 定态
2. 轨道
3. 跃迁 

德布罗意关系： $E=\hbar \omega$

### 波粒二象性

对于自由量子，使得波速度与群速度相等，则导致坐标与动量不对易关系

$$\Delta x\Delta p \geqslant \frac{\hbar}{2}$$

### 物质波

$|\psi(x,t)|^{2}$:量子出现在$x$处的几率，几率本身受因果关系支配

归一化：

$$\int^{\infty}_{\infty}|\psi(x,t)|^{2}dx=1$$


$\psi(x,t)e^{i\theta}\rightarrow\textbf{Projection Ray}$

思考：惠勒延迟选择实验

## 波动方程

$$\frac{\partial ^{2}}{\partial x^{2}}\vec{E}(x,t)-\frac{1}{c^{2}}\frac{\partial^{2}}{\partial t^{2}}\vec{E}(x,t)=0\\
\vec{E}\rightarrow \psi(x,t)\\
c\rightarrow v$$

推出 $\omega=kv$，存在一些困难，但实际上对于有质量的标量场满足，叫做Klein-Gordn方程

带入平面波解 $\frac{\partial^{2}}{\partial x^{2}}\psi(x,t)=D\frac{\partial \psi(x,t)}{\partial t}$

$$D=\frac{k^{2}}{i \omega}=\frac{2m}{i \hbar}$$

得薛定谔方程

$$i\hbar\frac{\partial }{\partial t}\psi(x,t)=\left[-\frac{\hbar^{2}}{2m}\frac{\partial ^{2}}{\partial x^{2}}+V(x,t)\right]\psi(x,t)$$

$$\implies i\hbar\frac{\partial }{\partial t}\ket{\psi}=\widehat{H}\ket{\psi}$$

动量算符： $p_{x}\rightarrow \widehat{p_{x}}=i\hbar\frac{\partial}{\partial x}$

概率密度守恒

$$\frac{\partial }{\partial t}\rho(x,t)+\frac{\partial j(x,t)}{\partial x}=0$$

波函数皆可表示为: $\psi(x,t)=\sqrt{\rho(x,t)}e^{\frac{1}{\hbar}S(x,t)}$

$$\vec{j}(x,t)=\frac{\rho(x,t)}{m}\left(\vec{\nabla}S(x,t)\right)$$

### 观测假设

**物理可观测量=算符**

实验观测值= $\braket{\psi|\vec{O}|\psi}=\braket{\vec{O}}$

$$\int \psi^{\dagger}\vec{O}\psi dx,\braket{\vec{O}}\in \text{Re}, \text{iff} \quad\vec{O}^{\dagger}=\vec{O}$$


> **经典仪器：遵从经典规律，可测量变化，可重复$\rightarrow$仪器测量体系状态，物理测量可重复性**

## 算符

$$\vec{A}^{\dagger}=\vec{A},\vec{A}\psi_{i}=a_{i}\phi_{i}\rightarrow \{\phi_{i}\}$$

对易子：

$$[\vec{A},\vec{B}]=\vec{A}\vec{B}-\vec{B}\vec{A}$$


Little Group $_1SO_2/E_{2}$