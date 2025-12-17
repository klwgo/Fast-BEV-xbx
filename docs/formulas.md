# 1.docx 公式与参数说明（Typora 版）

> 说明：  
> - 行内数学统一使用 `$...$`，块级公式统一使用 `$$...$$`；  
> - 所有变量名（如 $P$、$Q_i$、$\lambda_t$、$\Delta p$ 等）都写成 LaTeX 形式；  
> - 你只需要把整个文件保存为 `.md` 即可在 Typora 中正常渲染。

---

## 2. LiDAR 点云集合定义（第 2 条）

**公式：**

$$
P = \{ p_i \mid i = 1, 2, \dots, N \},\qquad
p_i = (x_i, y_i, z_i, r_i)
$$

其中：

- $P$：LiDAR 点云集合；
- $N$：集合 $P$ 中元素个数，为大于 0 的正整数；
- $p_i$：集合 $P$ 中的第 $i$ 个元素；
- $x_i, y_i, z_i$：点 $p_i$ 在三维空间中的坐标分量；
- $r_i$：点 $p_i$ 的某个标量属性。

---

## 3. 空间–语义双通道注意力（第 3 条）

**公式（语义 + 空间注意力）：**

$$
\alpha_{ij}
=
\operatorname{softmax}_j
\left(
\frac{Q_i K_j^\top}{\sqrt{d}}
+
\lambda \,\mathrm{SpatialDist}(v_i, v_j)
\right)
$$

其中：

- $v_i, v_j$：3D 顶点集中任意两个顶点；
- $Q_i$：顶点 $v_i$ 的语义特征向量（Query）；
- $K_j$：顶点 $v_j$ 的语义特征向量（Key）；
- $d$：语义特征维度（一般取 $256$），用于 $1/\sqrt d$ 缩放；
- $\mathrm{SpatialDist}(v_i, v_j)$：顶点 $v_i$ 和 $v_j$ 之间的 3D 空间距离；
- $\lambda$：可学习参数，初始值为 $1.0$，在训练中自适应调整；
- $\alpha_{ij}$：softmax 归一化后，表示顶点 $v_i$ 对顶点 $v_j$ 的综合注意力权重，用于刻画二者的语义相似度与空间关系。

---

## 4. 可学习阈值的超边构建（第 4 条）

**公式（超边集合与构建示意）：**

$$
E = \{ e_k \}_{k=1}^{L}
$$

$$
e_k
=
\{ v_i \}
\cup
\bigl\{
v_j
\ \big|\ 
\alpha_{ij} > \tau,\ j \in \mathcal{N}_{K}(i)
\bigr\}
$$

其中：

- $E$：最终构建出的超边集合；
- $e_k$：集合 $E$ 中的第 $k$ 条超边；
- $v_i$：超边 $e_k$ 中的一个“中心”顶点；
- $v_j$：与 $v_i$ 通过注意力连接的顶点；
- $\alpha_{ij}$：顶点 $i$ 与其他顶点的注意力分数（由第 3 条注意力公式计算）；
- $\mathcal{N}_K(i)$：顶点 $i$ 的 Top‑$K$ 注意力邻居集合；
- $K = 5$：限制每个顶点的最大连接数为 5；
- $\tau$：注意力阈值，初始值为 $0.5$，训练中可学习调整；
- “超边至少包含 2 个顶点”：意味着每条超边中至少有中心点和一个邻居点。

---

## 5. 语义相似度打分（第 5 条）

**公式（语义 + 空间打分）：**

$$
\alpha_{ij}
=softmax(
\frac{Q_i K_j^\top}{\sqrt{d}}
+
\lambda\,\mathrm{SpatialDist}(v_i, v_j))
$$

其中：

- $Q_i, K_j$：顶点 $v_i, v_j$ 的语义特征向量（与第 3 条一致）；
- $\mathrm{SpatialDist}(v_i, v_j)$：3D 空间距离；
- $\lambda$：可学习参数，初始值为 $1.0$，训练中自适应调整；
- $\alpha_{ij}$：综合语义特征与空间距离得到的打分，再送入 softmax 得到注意力权重。

---

## 6. 超图卷积（顶点级）基本公式（第 6 条）

**公式：**

$$
X'
=
\sigma\!\left(
D_v^{-1}
H
D_e^{-1}
H^\top
D_v^{-1}
X \Theta
\right)
$$

其中：

- $X \in \mathbb{R}^{N \times d}$：输入顶点特征，$N$ 为顶点数，$d$ 为特征维度；
- $X' \in \mathbb{R}^{N \times d'}$：输出顶点特征；
- $H \in \{0,1\}^{N \times L}$：顶点–超边关联矩阵，$L$ 为超边数量；
- $D_v$：顶点度对角矩阵，其对角元素为各顶点参与的超边数量；
- $D_e$：超边度对角矩阵，其对角元素为各超边包含的顶点数量；
- $\Theta$：可学习的线性变换矩阵；
- $\sigma$：非线性激活函数（如 ReLU）。

---

##  归一化关联矩阵与残差信息（第 7 条）

文字说明（无显式公式）：

- 先根据关联矩阵 $H$ 计算归一化算子  
  $D_v^{-\tfrac{1}{2}} H D_e^{-1} H^\top D_v^{-\tfrac{1}{2}}$；
- 再将顶点特征 $X$ 通过线性变换 $\Theta$ 映射到新特征空间；
- 然后通过超图卷积进行消息聚合；
- 最后将卷积结果与通过线性变换后的原始特征相加，形成残差连接，从而“保留原始信息，避免过度平滑”。

---

## 8. 融合特征与 BEV 空间参数（第 8 条）

- 多模态融合后的特征维度为 $[(M+N), 256]$；  
- 对应的 3D 坐标维度为 $[(M+N), 3]$；  
- BEV 空间参数范围：
  - $X \in [-50\text{m}, 50\text{m}]$；
  - $Y \in [-50\text{m}, 50\text{m}]$；
  - 分辨率（网格尺寸）为 $0.5\text{m}$。

---

## 9. BEV 网格尺寸计算（第 9 条）

**公式：**

$$
N_x
=
\frac{X_{\max} - X_{\min}}{\text{grid\_size}},
\qquad
N_y
=
\frac{Y_{\max} - Y_{\min}}{\text{grid\_size}}
$$

其中：

- $\text{grid\_size}$：网格尺寸（单位：米），文中为 $0.5\text{m}$；
- $X_{\min} = -50\text{m}$、$X_{\max} = 50\text{m}$：BEV 空间 $X$ 轴范围；
- $Y_{\min} = -50\text{m}$、$Y_{\max} = 50\text{m}$：BEV 空间 $Y$ 轴范围；
- 代入数值可得：
  - $N_x = 200$；
  - $N_y = 200$；
  - 总网格数 $N = N_x \times N_y = 40000$。

---

## 11. BEV 网格顶点集合初始化（第 11 条）

**公式：**

$$
G_{u,v} = \varnothing
$$

$$
G_{u,v}
=
\bigl\{
(i, f_i)
\ \big|\ 
(u_i, v_i) = (u, v)
\bigr\}
$$

其中：

- $G_{u,v}$：BEV 网格坐标 $(u, v)$ 对应的顶点集合；
- 初始化时 $G_{u,v} = \varnothing$：表示该网格尚无顶点；
- $i$：落入网格 $(u,v)$ 的顶点索引；
- $f_i$：顶点 $i$ 的特征向量（维度通常为 $256$）；
- $(u_i, v_i)$：顶点 $p_i$ 对应的 BEV 网格坐标。

---

## . BEV 超图构建的 MLP 结构（第 12 条）

**公式（结构）：**

$$
\mathrm{MLP}(x)
=
W_2 \,\sigma(W_1 x + b_1) + b_2
$$

其中：

- $\mathrm{MLP}(\cdot)$：用于 BEV 超图构建的多层感知机；
- 输入维度：$(k, 256)$，其中  
  - $k$ 为网格内顶点数量；  
  - $256$ 为顶点特征维度；
- 隐藏层维度：$64$；
- 输出维度：$(k, 1)$，为每个顶点输出一个标量权重；
- $W_1, W_2$：线性变换矩阵；
- $b_1, b_2$：偏置向量；
- $\sigma$：非线性激活函数（如 ReLU）。

---

## 13. 网格特征加权聚合（第 13 条）

**公式：**

$$
f_{\text{grid}}({u,v})
=
\sum_{i=1}^{k} w_i f_i,
\qquad
\sum_{i=1}^{k} w_i = 1
$$

其中：

- $f_{\text{grid}}({u,v})$：网格 $(u,v)$ 的聚合特征；
- $f_i$：网格内第 $i$ 个顶点的特征向量；
- $k$：该网格内顶点数；
- $w_i$：归一化注意力权重，满足 $\sum_{i=1}^{k} w_i = 1$；

---

## 14. 网格特征 L2 归一化（第 14 条）

**公式：**
$$
\tilde{f}_{\text{grid}}({u,v})
=
\frac{
f_{\text{grid}}({u,v})
}{
\left\|f_{\text{grid}}({u,v})\right\|_2 + \varepsilon
}
$$

其中：

- $\tilde{f}_{\text{grid}}({u,v})$：归一化后的网格特征；
- $f_{\text{grid}}({u,v})$：原始网格特征（由上一条公式计算）；
- $\|\cdot\|_2$：L2 范数；
- $\varepsilon$：数值稳定项（如 $10^{-6}$），用于避免分母为 0。

---

## 15. BEV 超图卷积（第 15 条）

**公式：**

$$
X'
=
\sigma\!\left(
D_v^{-\frac{1}{2}}
H
W_e
D_e^{-1}
H^\top
D_v^{-\frac{1}{2}}
X \Theta
\right)
$$

其中：

- $X \in \mathbb{R}^{N \times d}$：输入 BEV 顶点特征（$d = 256$）；
- $X'$：输出 BEV 特征；
- $H$：BEV 超图的顶点–超边关联矩阵；
- $D_v, D_e$：顶点 / 超边度对角矩阵；
- $W_e$：超边权重对角矩阵；
- $\Theta$：可学习变换矩阵；
- $\sigma$：非线性激活函数（ReLU）。

---

## 16. 空间邻接超边（第 16 条）

**构建规则：**
$$
e_{\text{spatial}}
=
\{(u,v)\}
\cup
\{ (u', v') \mid (u', v') \in N_8(u,v) \}
$$

其中：

- $e_{\text{spatial}}$：以网格 $(u,v)$ 为中心的空间邻接超边；
- $N_8(u,v)$：网格 $(u,v)$ 的 8 邻域（上、下、左、右以及四个对角方向的相邻网格）；
- 每个这样的超边连接一个中心网格及其 8 邻域，用于建模局部空间连续性。

---

## . 语义关联超边（第 17 条）

**距离计算公式：**

$$
d\bigl((u,v), (u',v')\bigr)
=
\left\|
p_{u,v} - p_{u',v'}
\right\|_2
$$

**超边构建公式（示意）：**

$$
S_{c}
=
\bigl\{
(u,v)
\;\big|\;
c_{u,v} = c,\ 
d\bigl((u,v), (u',v')\bigr) < D_{\text{semantic}}
\bigr\}
$$

其中：

- $c_{u,v} \in \{1,2,\dots,C\}$：来自 3D 空间感知模块的语义标签；
- $p_{u,v}$：网格 $(u,v)$ 对应的空间位置；
- $d((u,v),(u',v'))$：两个语义相同网格之间的欧氏距离；
- $D_{\text{semantic}} = 10\,\text{m}$：语义超边的距离阈值；
- $S_{c}$：类别为 $c$ 的语义关联超边，连接同类语义且空间接近的网格。

---

## 18. 动态交互超边与运动预测（第 18 条）

###  运动预测模型（LSTM）

**公式（示意）：**

$$
\Delta p
=
(\Delta x, \Delta y)
=
\mathrm{LSTM}\bigl(\{ F_{t-k} \}_{k=0}^{K} \bigr)
$$

其中：

- $\Delta p$：预测的 BEV 平面位移向量；
- $\Delta x, \Delta y$：位移在 $x$、$y$ 方向上的分量；
- $F_{t-k}$：历史 $K$ 帧的 BEV 特征。

### 18.2 当前与预测位置

**世界坐标：**
$$
p_{\text{curr}}
=
(x_{\min} + u \cdot \text{grid\_size},\ 
 y_{\min} + v \cdot \text{grid\_size})
$$

**预测位置：**

$$
p_{\text{pred}}
=
p_{\text{curr}} + \Delta p
$$

**预测网格坐标：**

$$
u'
=
\left\lfloor
\frac{x_{\text{pred}} - X_{\min}}{\text{grid\_size}}
\right\rfloor,
\qquad
v'
=
\left\lfloor
\frac{y_{\text{pred}} - Y_{\min}}{\text{grid\_size}}
\right\rfloor
$$

### 18.3 动态交互超边与权重

**超边：**

$$
e^{\text{motion}}_{u,v}
=
\{ (u,v), (u', v') \}
$$

**权重：**

- 动态交互超边权重直接使用运动预测输出的权重 $w_j$。

---

## 19. BEV 超图构建与卷积增强（第 19 条）

### 19.1 三类超边合并与去重

**合并：**

$$
\varepsilon_{\text{all}}
=
\varepsilon_{\text{spatial}}
\cup
\varepsilon_{\text{semantic}}
\cup
\varepsilon_{\text{motion}}
$$

**去重：**

​								$ε_{unique}$={e![img](file:////tmp/wps-will/ksohtml/wpsS90Y1o.jpg)e∈$ε_{all}$,ve'∈$ε_{all}$\{e},e≠e'}

其中：

- $\varepsilon_{\text{spatial}}$、$\varepsilon_{\text{semantic}}$、$\varepsilon_{\text{motion}}$：三类超边集合；
- $\varepsilon_{\text{all}}$：合并后的超边集合；
- $\varepsilon_{\text{unique}}$：去重后的超边集合，用于构建最终 BEV 超图。

### 19.2 关联矩阵构建

**线性索引：**

$$
i = v \cdot N_x + u
$$

**关联矩阵定义：**

$$
H_{i,e}
=
\begin{cases}
1, & \text{若网格 } i \text{ 属于超边 } e \\
0, & \text{否则}
\end{cases}
$$

其中：

- $i$：网格 $(u,v)$ 的线性索引；
- $N_x$：BEV 在 $x$ 方向的网格数；
- $H \in \{0,1\}^{N \times L}$：稀疏关联矩阵（COO 格式存储三元组 (行索引, 列索引, 值)）。

###  超边重要性与权重矩阵

**权重矩阵：**

$$
W_e
=
\mathrm{diag}(w_1, w_2, \dots, w_L)
$$

其中：

- 对动态交互超边，直接使用运动预测权重 $w_j$；
- 其他超边权重基于超边内网格特征相似度计算；
- $W_e$：对角阵形式的超边权重矩阵。

### 19.4 BEV 超图卷积与残差

**顶点度矩阵：**

$$
D_v = \mathrm{diag}(H \mathbf{1})
$$

**超边度矩阵：**

$$
D_e = \mathrm{diag}(H^\top \mathbf{1})
$$

**标准化超图卷积：**

$$
X' =
\sigma\!\left(
D_v^{-\frac{1}{2}}
H
W_e
D_e^{-1}
H^\top
D_v^{-\frac{1}{2}}
X \Theta
\right)
$$

**残差连接：**

$$
X^{(l+1)}
=
\mathrm{HyperConv}\bigl(X^{(l)}\bigr)
+
W_r X^{(l)}
$$

其中：

- $X \in \mathbb{R}^{N \times d}$：输入 BEV 特征（$d = 256$）；
- $\Theta \in \mathbb{R}^{d \times d'}$：可学习变换矩阵；
- $\sigma$：ReLU 激活函数；
- $W_r$：残差投影矩阵。

---

## 20. 时序特征对齐与时空超图（第 20 条）

###  时序缓存与运动补偿

**当前时刻 BEV 特征：**

$$
F_t \in \mathbb{R}^{H \times W \times 256}
$$

**历史特征：**

$$
F_{t-1}, F_{t-2}, \dots, F_{t-T},
\qquad T = 10
$$

**环形缓存更新：**

$$
\mathrm{Buffer}[t \bmod T] = F_t
$$

**运动补偿对齐（示意）：**

累积位移：

$$
\Delta p_{t-k \rightarrow t}
=
\sum_{m=t-k}^{t-1}
\Delta p_{m \rightarrow m+1}
$$

双线性插值（亚像素对齐）：

$$
F_{t-k \rightarrow t}^{u,v}
=
\sum_{(i,j)\in N_4(u',v')}
w_{ij} \, F_{t-k}^{i,j}
$$

其中：

- $N_4(u',v')$：4 邻域（上下左右）；
- $w_{ij}$：插值权重，$\sum w_{ij} = 1$。

### 20.2 时空顶点与关联矩阵

**顶点集合：**

$$
V^{\text{st}}
=
\{ (u,v,t) \mid 0 \le u < N_x,\ 0 \le v < N_y,\ 0 \le t < T \}
$$

**顶点总数：**

$$
N_{\text{st}} = N_x \cdot N_y \cdot T
$$

**线性索引（示意）：**

$$
n = t \cdot N + i,\qquad N = N_x N_y
$$

**时空关联矩阵：**

$$
H^{\text{st}} \in \{0,1\}^{N_{\text{st}} \times L_{\text{st}}}
$$

其中：

- $L_{\text{st}}$：时空超边数量。

### 20.3 三类时空超边

1. **空间超边（同一时刻内）：**

$$
e^{\text{spatial}}_{t,(u,v)}
=
\{ (u,v,t) \}
\cup
\{ (u',v',t) \mid (u',v') \in N_8(u,v) \}
$$

2. **时间超边（同一网格跨时刻）：**

$$
e^{\text{time}}_{u,v}
=
\{ (u,v,t) \mid t = 0,1,\dots,T-1 \}
$$

3. **时空超边（跨时刻空间关联）：**

$$
e^{\text{st}} =
\{ (u,v,t),\ (u',v',t') \mid \text{语义关联关系成立} \}
$$

其中，“语义关联关系”指同类物体网格（语义标签相同或同一实例）。

### 20.4 时空超图卷积与权重设计

**超图卷积：**

$$
X^{(l+1)}
=
\sigma\!\left(
D_v^{-\frac{1}{2}}
H
W_e
D_e^{-1}
H^\top
D_v^{-\frac{1}{2}}
X^{(l)} \Theta^{(l)}
\right)
$$

其中：

- $X^{(l)} \in \mathbb{R}^{N \times T \times d}$：第 $l$ 层顶点特征；
- $H \in \{0,1\}^{N \times T \times L}$：时空关联矩阵；
- $D_v$、$D_e$、$W_e$、$\Theta^{(l)}$：与前述类似，只是扩展到时序维度。

**时间超边权重：**

$$
w_e^{\text{time}}
=
\alpha \cdot
\mathrm{sim}\bigl(f_i^t, f_i^{t-1}\bigr)
$$

其中，$\alpha$ 为可学习缩放参数，$\mathrm{sim}$ 为余弦相似度。

**时空超边权重：**

$$
w_e^{\text{st}}
=
\beta \cdot
\exp\!\left(
-\frac{\|\Delta p\|^2}{\gamma^2}
\right)
$$

其中：

- $\beta, \gamma$：可学习参数；
- $\Delta p$：运动位移。

**门控机制：**

$$
G
=
\sigma\bigl(
W_g [X_t; X_{t-1}]
\bigr)
$$

其中：

- $X_t$：当前时刻特征；
- $X_{t-1}$：历史时刻特征；
- $[X_t; X_{t-1}]$：特征拼接；
- $W_g$：门控权重矩阵；
- $G$：门控信号；
- $\sigma$：Sigmoid 函数。

---

## 21. 多尺度 / 多模态 BEV 融合与分层超图（第 21 条）

### 21.1 多尺度特征统一到 BEV 分辨率

**公式：**

$$
F_i^{\text{bev}}
=
\operatorname{Conv}_{3\times3}
\bigl(
\mathrm{BilinearUpsample}_{s_i}(F_i)
\bigr)
$$

其中：

- $F_i$：第 $i$ 个尺度的特征图；
- $s_i$：上采样倍率；
- $\mathrm{BilinearUpsample}_{s_i}$：倍率为 $s_i$ 的双线性插值上采样算子；
- $\operatorname{Conv}_{3\times3}$：$3\times3$ 卷积。

### 21.2 跨模态特征融合

**多模态特征集合：**

$$
\mathcal{F}
=
\{ F_i^{\text{bev}} \} \cup \{ F_{\text{hyper}} \} \cup \cdots
$$

**自适应加权融合：**

$$
F_{\text{agg}}
=
\sum_{F \in \mathcal{F}}
\alpha_F F
$$

**权重计算：**

$$
h_F = \mathrm{GAP}(F)
$$

$$
\alpha_F
=
\frac{
\exp(q^\top W_F h_F)
}{
\sum_{F' \in \mathcal{F}}
\exp(q^\top W_F h_{F'})
}
$$

其中：

- $\mathcal{F}$：多模态特征集合；
- $F_{\text{agg}}$：融合后的特征；
- $\alpha_F$：特征 $F$ 的注意力权重；
- $h_F = \mathrm{GAP}(F)$：全局平均池化得到的通道描述向量；
- $W_F$：线性变换矩阵；
- $q$：可学习查询向量。

### 21.3 分层超图聚合

**公式：**

$$
F_{\text{HG}}
=
\sigma\!\left(
D_v^{-\frac{1}{2}}
H
W_e
D_e^{-1}
H^\top
D_v^{-\frac{1}{2}}
F_{\text{agg}} \Theta
\right)
$$

其中：

- $F_{\text{agg}}$：融合特征；
- $F_{\text{HG}}$：分层超图聚合后的特征；
- $H$：分层超图的关联矩阵（底层 BEV 网格，中层通道，高层任务组件）。

---

## 22. 自适应分发权重与梯度优化（第 22 条）

### 22.1 任务感知权重生成

**任务集合：**

$$
T = \{ t_1, t_2, \dots, t_{|T|} \}
$$

**任务嵌入：**

$$
e_t \in \mathbb{R}^{d_e},\qquad d_e = 64
$$

**任务权重计算：**

$$
w_t
=
\frac{
\exp\bigl(\mathrm{MLP}_t(e_t)\bigr)
}{
\sum_{t' \in T}
\exp\bigl(\mathrm{MLP}_{t'}(e_{t'})\bigr)
}
$$

**任务特定 MLP：**

$$
\mathrm{MLP}_t(x)
=
W_{2,t}\,\sigma(W_{1,t} x + b_{1,t}) + b_{2,t}
$$

### 22.2 空间与通道自适应权重

**空间注意力：**

$$
A_{\text{spatial}}
=
\operatorname{softmax}
\bigl(
\operatorname{Conv}_{3\times3}(F_{\text{HG}})
\bigr),\qquad
A_{\text{spatial}} \in \mathbb{R}^{H \times W}
$$

**通道注意力：**

$$
A_{\text{channel}}
=
\sigma\bigl(
\mathrm{FC}(\mathrm{GAP}(F_{\text{HG}}))
\bigr),\qquad
A_{\text{channel}} \in \mathbb{R}^{256}
$$

其中：

- $A_{\text{spatial}}$：空间权重图；
- $A_{\text{channel}}$：通道权重向量。

### 22.3 任务–空间–通道三重权重

**综合权重：**

$$
W_t
=
w_t \cdot A_{\text{spatial}} \cdot A_{\text{channel}}
$$

其中：

- $W_t \in \mathbb{R}^{H \times W \times 256}$：任务 $t$ 的完整权重张量；
- $w_t$：任务级标量权重；
- $A_{\text{spatial}}$、$A_{\text{channel}}$：空间 / 通道权重。

### 22.4 梯度归一化与自适应任务权重

**梯度归一化：**

$$
\nabla_{\theta} \mathcal{L}_{\text{total}}
=
\sum_{t \in T}
\lambda_t
\frac{
\nabla_{\theta} \mathcal{L}_t
}{
\left\|
\nabla_{\theta} \mathcal{L}_t
\right\|_2
+
\varepsilon
}
$$

**自适应权重调整：**

$$
\lambda_t
=
\frac{
\exp(\beta_t / \mathcal{L}_t)
}{
\sum_{t' \in T}
\exp(\beta_{t'} / \mathcal{L}_{t'})
}
$$

其中：

- $\mathcal{L}_{\text{total}}$：多任务总损失；
- $\mathcal{L}_t$：任务 $t$ 的损失；
- $\lambda_t$：任务 $t$ 的权重；
- $\beta_t$：任务 $t$ 的温度参数，用于控制权重对损失变化的敏感度；
- $\varepsilon$：数值稳定项。
