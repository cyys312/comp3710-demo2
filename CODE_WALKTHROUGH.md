# 代码讲解 — 每个文件在干嘛

> **在线合并版（推荐）**：<https://claude.ai/code/artifact/2b865fba-d70b-4ef0-8081-dc3f70640469>
> 每个 Part 下并排放问答与代码讲解，代码带语法高亮、侧栏可跳转、检查清单可勾选。
> 离线副本：[`docs/demo_handbook.html`](docs/demo_handbook.html)，双击即可用浏览器打开。

> 配套 [`DEMO_PREP.md`](DEMO_PREP.md) 使用。
> 那份是「被问到 X 怎么答」，这份是「老师指着代码问『这段在干嘛』怎么答」。
>
> 全部 shape 与参数量都是实际跑出来的，不是估算。

---

## 0. 三十秒讲清整个仓库

四个 Part，共 2479 行 Python。目录按 Part 分，`recognition/` 下的三个子项目
用课程约定的四件套：

| 文件 | 职责 |
|---|---|
| `modules.py` | 网络与损失的定义。**只定义，不训练** |
| `dataset.py` | 数据加载与预处理。**只读数据，不碰模型** |
| `train.py` | 训练循环、验证、存 checkpoint、画曲线 |
| `predict.py` | 载入 checkpoint 做推理与可视化（**demo 现场跑这个**） |

这样拆的好处：`predict.py` 不需要知道怎么训练，`modules.py` 不需要知道数据从哪来。
演示时老师要看推理，直接跑 `predict.py`，它只依赖 `modules.py` 和一个 checkpoint。

---

## 1. 三处跨项目的共用设计

这三个函数在多个文件里重复出现，先讲一次，后面不再重复。

### `pick_device()` — 一份代码两个环境

```python
def pick_device() -> torch.device:
    """优先 CUDA（Rangpur A100），其次 Apple MPS，最后退回 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

同一份代码在 Mac（Apple MPS）和集群（CUDA A100）上都能跑，不用改任何东西。

### `auto_workers()` — 默认值不能照抄

```python
def auto_workers(requested: int, device: torch.device) -> int:
    if requested >= 0:
        return requested
    return 4 if device.type == "cuda" else 0
```

**为什么 MPS 上要用 0 个 worker** —— 这是实测出来的反直觉结论：

| 配置 | 耗时 |
|---|---|
| `num_workers=0` | **9 ms/batch** |
| `num_workers=6, persistent=True` | 262 ms/batch |

单张 OASIS 切片解码只要约 1 ms，多进程的 IPC 序列化开销远大于收益，**慢 29 倍**。
集群上 CPU 核多、数据在网络盘，多 worker 才划算。

### `sync()` — GPU 计时的坑（Part 1 用）

```python
def sync(device: torch.device) -> None:
    """GPU 上的算子是异步下发的，计时前必须同步，否则测到的只是下发时间。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
```

**这行不加，Part 1 的所有 GPU 计时都是错的** —— 会测出个接近 0 的数，
因为 Python 只是把 kernel 丢进队列就返回了。

---

## 2. Part 1 — `part1_dft/dft.py`（317 行）

### 这个文件在干嘛

两件事：(1) 用奇次谐波的傅里叶级数重建方波；(2) 实现四种 DFT 并比较耗时。

### 四种 DFT 实现，从慢到快

**① 教科书式双重循环** — 严格 O(N²)，只为了对照

```python
def naive_dft_loops(x):
    n_samples = len(x)
    out = np.zeros(n_samples, dtype=np.complex128)
    for k in range(n_samples):          # 每个频率 bin
        for n in range(n_samples):      # 累加所有采样点的贡献
            out[k] += x[n] * np.exp(-2j * np.pi * k * n / n_samples)
    return out
```

直接照抄 DFT 的定义式 `X[k] = Σₙ x[n]·e^(-2πikn/N)`。

**② NumPy 矩阵形式** — 同样 O(N²)，但快 25 倍

```python
def _dft_matrix_np(n_samples):
    idx = np.arange(n_samples)
    return np.exp(-2j * np.pi * np.outer(idx, idx) / n_samples)   # W[k,n]

def naive_dft(x):
    return _dft_matrix_np(len(x)) @ x
```

把两层循环写成一次矩阵-向量乘 `X = W·x`。`np.outer(idx, idx)` 生成的是
k·n 的外积矩阵。**乘加次数完全没变，变的只是谁在执行循环**——从 Python
解释器换成了 BLAS。

**③ PyTorch GPU 版** — 任务书明确要求的那个

```python
def naive_dft_torch(x):
    n_samples = x.shape[-1]
    idx = torch.arange(n_samples, device=x.device, dtype=torch.float32)
    angle = -2.0 * torch.pi * idx[:, None] * idx[None, :] / n_samples
    # 用 cos/sin 组装复数矩阵，避免依赖后端的复数指数算子
    w = torch.complex(torch.cos(angle), torch.sin(angle))
    return w @ x.to(torch.complex64)
```

三个要点，都可能被问：

- **`device=x.device`**：DFT 矩阵在输入所在的设备上构造。传 GPU 张量进来，
  整个计算就在 GPU 上完成——这就是「显式跑在 GPU 上」的实现方式。
- **`idx[:, None] * idx[None, :]`**：广播出 k·n 的外积，等价于 numpy 的 `outer`。
- **用 `cos`/`sin` 而不是复数 `exp`**：不同后端对复数指数的支持不一致，
  拆成实部虚部再用 `torch.complex` 组装，到哪都能跑。

**④ FFT** — `np.fft.fft` / `torch.fft.fft`，O(N log N)

### 计时函数里的两个细节

```python
def _time_it(fn, repeats, device=None):
    """跑 repeats 次取最小值（最小值比均值更抗系统噪声）。"""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device is not None:
            sync(device)                    # 关键：等 GPU 真正算完
        best = min(best, time.perf_counter() - start)
    return best
```

- **取最小值不取均值**：均值会被偶发的系统调度、GC 拉高；最小值更接近真实开销。
- **预热**：`benchmark()` 里正式计时前会先各跑一次，把 kernel 编译和显存分配的
  一次性开销排除掉。

### 老师可能指着问的

**`np.linspace(0.0, T, N, endpoint=False)` 里的 `endpoint=False`** ——
信号是周期的，`t=0` 和 `t=T` 是同一个相位点，两端都取会多采一个点导致频谱泄漏。

---

## 3. Part 2 — `part2_eigenfaces/eigenfaces.py`（159 行）

### 这个文件在干嘛

LFW 人脸 → 去均值 → SVD 得 eigenfaces → 投影到 face space → 随机森林分类。

### PCA 的核心就这几行

```python
def compute_pca(X_train, X_test, n_components=N_COMPONENTS):
    mean = np.mean(X_train, axis=0)     # 均值只用训练集算
    X_train = X_train - mean
    X_test = X_test - mean              # 但同一个均值也减到测试集上

    U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    components = Vt[:n_components]      # 前 k 个右奇异向量 = 主成分

    X_train_pca = X_train @ components.T    # 投影到 face space
    X_test_pca = X_test @ components.T
    return components, S, X_train_pca, X_test_pca, mean
```

**这三行里藏着最容易被问的点**：

1. **`mean` 只由 `X_train` 算**，然后减到两边。若用全体数据算均值，
   测试集的信息就通过均值泄漏进了模型。SVD 同理，只在训练集上做。
2. **`Vt[:n_components]`** —— 对去均值的 X 做 SVD 得 `X = U S Vᵀ`，
   则协方差 `XᵀX = V S² Vᵀ`。所以 **V 的行就是协方差矩阵的特征向量**，
   奇异值平方正比于特征值。直接对 X 做 SVD 比先算 1850×1850 的协方差矩阵
   再做特征分解更稳定也更省。
3. **`full_matrices=False`** —— 只算需要的那部分奇异向量。
   样本数 966 < 特征数 1850，完整的 U 会是 966×966、V 是 1850×1850，没必要。

```python
eigenfaces = components.reshape((N_COMPONENTS, h, w))
```

主成分本身是 1850 维向量，reshape 回 50×37 就是能看的「特征脸」。

### 我自己加的两组对照实验

```python
def ablation(X_train_pca, X_test_pca, y_train, y_test):
    # 实验 1：主成分个数
    for k in (25, 50, 100, 150):
        clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train_pca[:, :k], y_train)
        acc_k = (clf.predict(X_test_pca[:, :k]) == y_test).mean()

    # 实验 2：类别加权
    for weight in (None, "balanced"):
        clf = RandomForestClassifier(..., class_weight=weight)
```

注意 **`X_train_pca[:, :k]`** —— SVD 只做一次，切前 k 列就是「只保留前 k 个成分」，
不用重新算。结果：50 个成分（0.6553）**优于** 150 个（0.5776）。

---

## 4. Part 3.1 — `part3_cnn/cnn_lfw.py`（117 行）

### 网络结构与实测 shape

```python
class SimpleCNN(nn.Module):
    def __init__(self, n_classes, in_shape):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 任务书要求
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # 两层 3x3 / 32 filters
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

        # 用一次假前向推出展平后的维度，避免手算
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *in_shape)
            flat_dim = self._features(dummy).flatten(1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, n_classes)
```

| 阶段 | shape |
|---|---|
| 输入 | (B, 1, 50, 37) |
| 两层卷积 + 两次池化 | (B, 32, 12, 9) |
| 展平 | (B, 3456) |
| 输出 logits | (B, 7) |

**那段「假前向」值得讲**：50×37 经两次 2×2 池化不是整除（50→25→12，37→18→9），
手算容易错。跑一次全零张量让 PyTorch 自己算出 3456，改输入尺寸时不用改代码。

### 参数分布 — 一个很好的讲解点

| 层 | 参数量 | 占比 |
|---|---|---|
| conv1 | 320 | 0.07 % |
| conv2 | 9,248 | 2.0 % |
| **fc1** | **442,496** | **97.7 %** |
| fc2 | 903 | 0.2 % |
| 合计 | 452,967 | |

**97.7 % 的参数集中在第一个全连接层**。这正解释了为什么 `Dropout(0.5)`
放在 fc1 之后而不是卷积层之间——过拟合风险几乎全在那里。
卷积层参数共享，本身正则性就强。

---

## 5. Part 3.2 — `part3_cnn/dawnbench/`（443 行，4 分）

### `modules.py` — 为 32×32 改造的 ResNet-18

```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        self.in_planes = 64
        # 与 ImageNet 版的唯一差别就在下面这一行 + 没有 maxpool
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
```

实测 shape（输入 3×32×32）：

| 阶段 | shape |
|---|---|
| conv1 | (B, 64, 32, 32) ← **仍是 32×32，没被下采样** |
| layer1 | (B, 64, 32, 32) |
| layer2 | (B, 128, 16, 16) |
| layer3 | (B, 256, 8, 8) |
| layer4 | (B, 512, 4, 4) |
| linear | (B, 10) |

参数量 11.17 M。ImageNet 版首层是 7×7 stride=2 + maxpool stride=2，
进 layer1 时只剩 8×8——32×32 的输入经不起这个。

残差块的捷径：

```python
self.shortcut = nn.Sequential()          # 默认是恒等映射
if stride != 1 or in_planes != planes * self.expansion:
    self.shortcut = nn.Sequential(       # 尺寸或通道变了才用 1x1 卷积投影
        nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * self.expansion),
    )
```

**空的 `nn.Sequential()` 就是恒等映射**——只有当 stride 或通道数变化、
张量加不起来时才需要投影。

### `dataset.GPUCifar` — 这是拿到那 2 分的关键

**为什么需要它**：先测量，发现 torchvision 的 CPU 管线在 A100 上
227 秒跑不完一个 epoch（>2.3 秒/步），而模型本身一步只要几十毫秒——
瓶颈在 CPU 不在 GPU。

**核心思路**：CIFAR-10 全部像素只有 153 MB（uint8），一次性搬进 40 GB 显存。

```python
# (N,32,32,3) uint8 -> (N,3,32,32) uint8，一次性搬进显存
images = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
self.images = images.to(device, non_blocking=True)
# ToTensor 会把像素除以 255，这里省掉这一步，直接把 MEAN/STD 乘回 255
self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255.0
```

**随机裁剪的 GPU 实现**——这段最可能被指着问：

```python
def _random_crop(self, x):
    n = x.shape[0]
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")          # (N,3,40,40)
    offset_y = torch.randint(0, 9, (n,), device=dev)         # 每个样本独立的偏移
    offset_x = torch.randint(0, 9, (n,), device=dev)
    rows = offset_y.view(n, 1, 1) + torch.arange(32, device=dev).view(1, 32, 1)
    cols = offset_x.view(n, 1, 1) + torch.arange(32, device=dev).view(1, 1, 32)
    batch_idx = torch.arange(n, device=dev).view(n, 1, 1)
    # 混用高级索引与切片时，高级索引的维度会排到最前面 -> (N,32,32,3)
    cropped = padded[batch_idx, :, rows, cols]
    return cropped.permute(0, 3, 1, 2).contiguous()
```

逐行拆解：

1. `F.pad(..., mode="reflect")` — 补到 40×40，与 `T.RandomCrop(32, padding=4,
   padding_mode="reflect")` 的第一步一致。
2. `torch.randint(0, 9, (n,))` — 偏移范围 [0, 8]，因为 40−32=8。
   **形状是 `(n,)` 而不是标量**，所以每个样本的偏移独立，和逐张做增强等价。
3. `rows` / `cols` — 广播出每个样本要取的 32 个行号和列号。
4. `padded[batch_idx, :, rows, cols]` — 高级索引。**混用高级索引与切片 `:` 时，
   高级索引的维度会被排到最前面**，所以结果是 (N,32,32,3) 而不是 (N,3,32,32)，
   必须再 `permute` 回来。这是最容易写错的一步。

**怎么验证它和 torchvision 等价**：把偏移固定成 (4,4)，输出必须与原图逐像素相同。
我做了这个测试并通过——它同时证明了第 4 步的维度顺序没搞反。

翻转：

```python
def _random_flip(self, x):
    do_flip = torch.rand(x.shape[0], device=x.device) < 0.5
    return torch.where(do_flip.view(-1, 1, 1, 1), x.flip(-1), x)
```

`torch.where` 按样本选择翻转版或原版，无分支、全并行。

**效果**：1.9 秒/epoch，提速 100 倍以上。

### `train.py` — 混合精度与演示保护

```python
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
...
with torch.amp.autocast("cuda", enabled=use_amp):
    loss = criterion(model(x), y)
scaler.scale(loss).backward()     # 损失放大，避免 fp16 梯度下溢
scaler.step(optimiser)            # 内部先把梯度缩回去，遇到 inf/nan 就跳过这步
scaler.update()                   # 动态调整缩放因子
```

```python
if acc > best_acc:
    best_acc = acc
    if not args.no_save:          # 演示跑单个 epoch 时不写 checkpoint
        torch.save(...)
```

**`--no-save` 是演示时的保护**：不加的话，那一个 epoch 的模型会覆盖掉
94.15 % 的 checkpoint（保存条件是 `acc > best_acc` 而 `best_acc` 从 0 开始）。

---

## 6. Task 1 — `recognition/vae_oasis/`（451 行）

### `modules.py` — 实测 shape（latent_dim=32, 128×128）

| 阶段 | shape |
|---|---|
| 输入 | (B, 1, 128, 128) |
| 编码器 4 次 stride=2 卷积 | (B, 256, 8, 8) |
| `fc_mu` / `fc_logvar` | (B, 32) / (B, 32) |
| 重参数化后的 z | (B, 32) |
| 解码器重建 | (B, 1, 128, 128)，sigmoid 后落在 [0,1] |

参数量 2.97 M（编码器 1.74 + 解码器 1.23）。

### 重参数化技巧 — 一定会被问

```python
@staticmethod
def reparameterise(mu, logvar):
    """重参数化技巧：z = mu + sigma * eps，让采样这一步可导。"""
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)
```

**为什么网络输出 `logvar` 而不是 `sigma`**：方差必须为正。
直接输出 sigma 需要额外约束；输出 log 方差则取值可以是整个实轴，
`exp(0.5 * logvar)` 出来自然为正。这也让数值更稳定。

**为什么这样就可导**：随机性被隔离在 `randn_like` 里，它与网络参数无关。
z 对 mu 和 std 是确定性的可导函数，梯度能正常回传到编码器。

### 损失函数

```python
def vae_loss(recon, x, mu, logvar, beta=1.0):
    """ELBO 的负值：重建项 (BCE) + beta * KL 项。"""
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl
```

- **`reduction="sum"` 再除以 batch** = 每张图上所有像素的 BCE 之和。
  所以数值在几千量级（128×128 = 16384 个像素）。
  用 `"mean"` 会把 KL 和重建的相对权重搞错。
- **KL 那一行**是两个高斯之间 KL 散度的解析解。q = N(mu, sigma²)、
  p = N(0, I) 时可以直接写成闭式，不需要采样估计。

### `predict.py` — 流形网格与两个自己写的工具

```python
def normal_ppf(q):
    """标准正态的分位数函数。等价于 scipy.stats.norm.ppf，但不依赖 scipy——
    集群 conda 环境里没装。ppf(q) = sqrt(2) * erfinv(2q - 1)。"""
    q = torch.as_tensor(np.asarray(q, dtype=np.float64))
    return (torch.sqrt(torch.tensor(2.0, dtype=torch.float64)) *
            torch.erfinv(2 * q - 1)).numpy()
```

实测与 scipy 的最大偏差 **8.882e-16**（双精度的噪声水平）。

```python
grid_x = normal_ppf(np.linspace(0.02, 0.98, n))     # 按分位数取，不是等距
```

**为什么用分位数**：先验是标准正态。等距取点会让网格边缘落在概率密度极低的区域，
那里模型训练时几乎没见过，解出来的图不可信。分位数采样让网格**均匀覆盖概率质量**。

```python
def pca_2d(latents):
    """UMAP 的兜底：集群没装 umap-learn，PCA 用 torch 的 SVD 就能做。"""
    centred = latents - latents.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(centred, full_matrices=False)
    projected = u[:, :2] * s[:2]
    ratio = (s ** 2 / (s ** 2).sum())[:2]
    return projected.numpy(), ratio.numpy()
```

`u[:, :2] * s[:2]` 就是数据在前两个主成分上的坐标。

---

## 7. Task 2 — `recognition/unet_oasis/`（474 行，重点）

### `modules.py` — 实测 shape（base=32, 输入 1×256×256）

| 阶段 | shape | 说明 |
|---|---|---|
| enc1 | (B, 32, 256, 256) | ↓ 这四个的输出要留着做 skip |
| enc2 | (B, 64, 128, 128) | |
| enc3 | (B, 128, 64, 64) | |
| enc4 | (B, 256, 32, 32) | |
| **bottleneck** | **(B, 512, 16, 16)** | 感受野最大，空间精度最低 |
| dec4 | (B, 256, 32, 32) | ↑ 这四个各拼一次 skip |
| dec3 | (B, 128, 64, 64) | |
| dec2 | (B, 64, 128, 128) | |
| dec1 | (B, 32, 256, 256) | |
| head | (B, 4, 256, 256) | 4 通道 logits |

参数量 7.76 M。

### skip connection 就是这四行

```python
def forward(self, x):
    e1 = self.enc1(x)
    e2 = self.enc2(self.pool(e1))
    e3 = self.enc3(self.pool(e2))
    e4 = self.enc4(self.pool(e3))
    b = self.bottleneck(self.pool(e4))

    d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # skip connection
    d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
    d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
    return self.head(d1)                                   # logits (B, C, H, W)
```

**`torch.cat([..., e4], dim=1)` 是沿通道维拼接**，不是相加。
`up4(b)` 输出 (B,256,32,32)，`e4` 也是 (B,256,32,32)，拼完是 (B,512,32,32)，
所以 `dec4 = DoubleConv(chs[3]*2, chs[3])` 的输入通道是 512。

**为什么必须有它**：编码器一路下采样，到 bottleneck 只剩 16×16——
知道「这是什么组织」但不知道「边界在哪」。skip 把编码器同尺度的高分辨率特征
直接接过来，补回被丢掉的空间细节。

### DSC 的两处修正 — 最能体现理解的部分

**修正 1：软 Dice 只能当损失，不能当指标**

```python
def dice_per_class(logits, target_one_hot, eps=1e-6):
    """**软** Dice：直接用 softmax 概率算，可导，专门给损失函数用。

    注意不要拿它当评测指标上报 —— 评分要求的 DSC 是对**硬预测**（argmax 之后
    的类别图）算的。软 Dice 在模型不自信时会偏低，在过分自信时又会偏高。
    """
    probs = F.softmax(logits, dim=1)
```

**修正 2：DSC 是比值，不能按 batch 平均**

```python
@torch.no_grad()
def hard_dice_counts(logits, target_one_hot):
    """返回逐类的 (交集, 基数) 计数，用于跨 batch 累加。

    为什么要返回计数而不是直接返回 DSC：DSC 是个比值，
    「先按 batch 算 DSC 再取平均」并不等于「整个数据集上的 DSC」，
    而且最后一个不满的 batch 会被赋予同等权重。正确做法是把分子分母
    分别累加完，最后再做一次除法。
    """
    num_classes = logits.shape[1]
    pred = F.one_hot(logits.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(pred * target_one_hot, dims)
    cardinality = torch.sum(pred + target_one_hot, dims)
    return intersection, cardinality
```

用起来是这样：

```python
inter = torch.zeros(NUM_CLASSES, device=device)
card = torch.zeros(NUM_CLASSES, device=device)
for x, y in loader:
    i, c = hard_dice_counts(model(x), to_one_hot(y, NUM_CLASSES))
    inter += i          # 分子分母分别累加
    card += c
return dice_from_counts(inter, card)    # 全部累完再除一次
```

数学上：`2ΣI / ΣC ≠ (1/n)·Σ(2Iᵢ/Cᵢ)`。

`dims = (0, 2, 3)` — 沿 batch、高、宽求和，**保留通道维**，所以结果是每类一个数。

### 组合损失

```python
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5):
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, logits, target_idx, target_one_hot):
        return self.ce_weight * self.ce(logits, target_idx) + \
               (1 - self.ce_weight) * dice_loss(logits, target_one_hot)
```

**注意它同时收两种标签**：`target_idx` 是 (B,H,W) 的类别索引给 CE 用，
`target_one_hot` 是 (B,4,H,W) 给 Dice 用。任务书要求 one-hot 输出，这里满足了。

### `dataset.py` — 标签解码

```python
LABEL_VALUES = np.array([0, 85, 170, 255], dtype=np.int16)
...
img = self._load(self.img_paths[idx]).astype(np.float32) / 255.0
seg = self._load(self.seg_paths[idx], nearest=True)   # 标签必须最近邻，不能插值
```

```python
def _load(self, path, nearest=False):
    img = Image.open(path).convert("L")
    if self.image_size:
        img = img.resize((self.image_size, self.image_size),
                         Image.NEAREST if nearest else Image.BILINEAR)
```

**`Image.NEAREST` 那个三元表达式是整个文件最关键的一行**：
标签灰度 {0, 85, 170, 255} 是**类别编号**不是连续量。双线性插值会在边界产生
42、127 这样的中间值，映射回类别时会凭空造出错误归属。原图是连续量，用双线性。

原图与标签的配对：

```python
seg_lookup = {p.name.replace("seg_", "", 1): p for p in self.seg_dir.glob("*.png")}
for p in self.img_paths:
    key = p.name.replace("case_", "", 1)
    if key not in seg_lookup:
        raise KeyError(f"{p.name} 找不到对应标签")
```

`case_441_slice_0.nii.png` 去掉前缀是 `441_slice_0.nii.png`，
`seg_441_slice_0.nii.png` 去掉前缀也是它——用这个当 key 配对。
**找不到就直接抛异常**，不静默跳过：数据错位是最难查的 bug。

---

## 8. Task 3 — `recognition/gan_oasis/`（518 行）

### 实测 shape（latent_dim=128, base=64, 128×128）

| 阶段 | shape |
|---|---|
| z | (B, 128) |
| 生成器每次上采样 | 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → **128×128** |
| 生成器输出 | (B, 1, 128, 128)，tanh 后落在 [−1, 1] |
| 判别器输出 | (B,)，**logit，未过 sigmoid** |

生成器 13.24 M，判别器 11.16 M。

**判别器为什么不过 sigmoid**：配 `BCEWithLogitsLoss` 使用。
它内部把 sigmoid 和 BCE 合并计算（log-sum-exp 技巧），数值上比
先 sigmoid 再 BCE 稳定得多。

### R1 梯度惩罚

```python
def r1_penalty(discriminator, real_images):
    """R1 梯度惩罚：判别器在**真实样本**处梯度的平方范数。"""
    real_images = real_images.detach().requires_grad_(True)
    logits = discriminator(real_images)
    grad = torch.autograd.grad(
        outputs=logits.sum(), inputs=real_images, create_graph=True)[0]
    return grad.pow(2).flatten(1).sum(1).mean()
```

逐行：

- **`detach().requires_grad_(True)`** — 我们要的是 logits 对**输入图像**的梯度，
  不是对参数的。先 detach 断开原来的图，再标记需要梯度。
- **`create_graph=True`** — 这个惩罚项本身还要再对参数求一次导（二阶导），
  所以求梯度的过程也必须建图。**这是它贵的原因，也是要惰性正则的原因。**
- **直觉**：在真实数据附近把判别器压平。判别器一旦过强，
  生成器收到的梯度就会消失，训练随即崩掉。

惰性正则的用法：

```python
if step % args.r1_every == 0:
    penalty = r1_penalty(disc, real)
    loss_d = loss_d + (args.r1_gamma / 2) * penalty * args.r1_every
```

**乘 `r1_every` 是为了补偿频率**：每 16 步做一次但强度乘 16，
平均下来与每步都做等价，算力却省了 15/16。

### 生成器 EMA

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for shadow_p, p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.lerp_(p.detach(), 1.0 - self.decay)
        # buffer（BatchNorm 的 running stats）直接拷贝，不做平均
        for shadow_b, b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(b)
```

- **`lerp_(p, 1-decay)`** 就是 `shadow = shadow*decay + p*(1-decay)`，
  用 in-place 的 lerp 省一次内存分配。
- **buffer 直接拷贝不做平均** —— BatchNorm 的 running_mean/var 本身
  已经是滑动平均了，再平均一次没有意义。

### 多样性指标 — 本任务的核心贡献

```python
@torch.no_grad()
def diversity_score(images, max_pairs=2048):
    """样本两两之间的平均 L2 距离 —— 用来量化 mode collapse。"""
    flat = images.flatten(1)
    n = flat.size(0)
    idx_a = torch.randint(0, n, (max_pairs,), device=flat.device)
    idx_b = torch.randint(0, n, (max_pairs,), device=flat.device)
    keep = idx_a != idx_b                       # 排除自己和自己配对
    return (flat[idx_a[keep]] - flat[idx_b[keep]]).norm(dim=1).mean().item()
```

**`keep = idx_a != idx_b` 这行不能少**：自己和自己的距离是 0，
混进去会把平均值拉低，让健康的模型看起来像崩塌了。

用随机配对而不是算全部 n(n−1)/2 对，是因为后者在 n 大时开销平方增长，
而随机 2048 对的估计已经足够稳定。

### 训练循环里的两处关键

```python
fake = gen(z).detach()          # detach：这一步不更新生成器
loss_d = (bce(logits_real, torch.full_like(logits_real, args.label_smooth))
          + bce(logits_fake, torch.zeros_like(logits_fake)))
```

**`.detach()` 必须有**：更新判别器时，梯度不能流回生成器。
忘了这个的话生成器会朝「帮判别器分辨」的方向更新，训练立刻毁掉。

```python
fake = gen(z)                   # 这次不 detach
loss_g = bce(disc(fake), torch.ones(batch, device=device))
```

**非饱和损失**：给假样本打上「真」的标签去算 BCE，等价于最大化 `log D(G(z))`。
原始 GAN 论文的形式是最小化 `log(1−D(G(z)))`，在训练早期判别器很强时
梯度几乎为 0——同一篇论文里就指出了这个问题并给出了这个替代形式。

---

## 9. 老师最可能指着问的十行

| 位置 | 那行代码 | 一句话答案 |
|---|---|---|
| `dft.py` | `sync(device)` | GPU 算子异步下发，不同步就只测到下发时间 |
| `dft.py` | `torch.complex(cos, sin)` | 不依赖后端的复数指数算子，到哪都能跑 |
| `eigenfaces.py` | `mean = np.mean(X_train, axis=0)` | 均值只用训练集算，否则测试集信息泄漏 |
| `cnn_lfw.py` | 假前向推 `flat_dim` | 50×37 两次池化不整除，手算易错 |
| `dawnbench/modules.py` | `nn.Conv2d(3, 64, 3, stride=1)` | 32×32 经不起 ImageNet 版的 4 倍下采样 |
| `dawnbench/dataset.py` | `padded[batch_idx, :, rows, cols]` | 高级索引维度排最前，结果要 permute 回来 |
| `vae/modules.py` | `torch.exp(0.5 * logvar)` | 输出 log 方差保证 sigma 恒正、数值更稳 |
| `unet/modules.py` | `hard_dice_counts` 返回计数 | DSC 是比值，batch 平均 ≠ 数据集级 |
| `unet/dataset.py` | `Image.NEAREST if nearest` | 标签是类别编号，插值会造出不存在的类别 |
| `gan/train.py` | `gen(z).detach()` | 更新判别器时梯度不能流回生成器 |

---

## 10. 如果被问「这段是不是 AI 写的」

诚实答：**代码是在 Claude Code 协助下写的，但每一处我都验证过并修正了它的错误**，
然后举 [`AI_PROMPTS.md`](AI_PROMPTS.md) 里的具体例子——尤其这三个，
因为它们说明我真的读懂了而不是照抄：

1. **AI 把软 Dice 同时当损失和评测指标** → 我拆成两个函数，
   因为评分要的 DSC 是对 argmax 之后的硬预测算的
2. **AI 按 batch 平均 DSC** → 比值不能这样平均，改成累加交集与基数
3. **AI 给的 slurm 参数三处全错** → `--mem` 会让作业永远 PENDING，
   我用 `sinfo`/`scontrol`/`sacctmgr` 逐条核对后才发现

再补一句自己做的：**Part 3.2 的 GPU 数据管线是我测出瓶颈后自己加的**，
等价性验证（偏移 (4,4) 时精确还原原图）也是自己设计的。
