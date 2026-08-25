# 超越函数的开销与精度

这一页记录 elementwise 算子里超越函数带来的开销与精度问题。它们不属于访存受限 —— 恰恰相反，只有当算术开始成为瓶颈时才需要考虑这些。

这一页的数字都是**访存带宽** —— 搬运的字节数（读一次加写一次）除以耗时，单位 TB/s。elementwise 算子本该是访存受限的，所以这个数离机器的访存上限（这块卡实测 4.07 TB/s）有多远，衡量的就是算术占掉了多少时间：越接近上限，说明算术越不再是瓶颈。

超越函数走 MUFU（special function unit），吞吐是 fp32 FMA 的 1/8 —— compute capability 9.0 上每 SM 每周期 16 个结果（对比 FMA 这个数字是 128）。一个 elementwise 算子里若含两三个精确超越函数，算术的耗时便不再可以忽略。

## 1. 在 fp32 上做超越函数与除法 {#fp32-transcendental}

MUFU 只有 fp32 的形式，窄类型的超越函数无论怎么写都要转到 fp32 再转回。差别在于这对转换落在哪里：写在存储 dtype 上，它落在**每一次运算的两侧**；写在 fp32 上，整个式子只在读入与写出各转一次。中间量的精度也随之不同 —— 留在 fp16 意味着每一步都被舍到 11 位有效位。

读生成的 CUDA 可以确认这一点：`x * T.sigmoid(x)` 写在 bf16 上，生成的代码里指数仍然是 `expf`，也就是仍然转到了 fp32，只是每次运算都转一遍。

**fp16 上没有选择。** `T.sigmoid` 或 `T.exp` 直接作用在 fp16 值上，生成的 CUDA 会落到 `hexp(half)`，而 CUDA 13.2 里没有匹配的重载，nvcc 直接报错。bf16 两种写法都能编译，所以下面的实测用 bf16。

### 实测

H200，SM 时钟锁在 1830 MHz，bf16 `silu`，输入 512 MB（大于 60 MiB 的 L2），读加写共 1 GB。精度是 fp32 参考值上的最大相对误差，输入覆盖 $[-30, 30]$：

| 写法 | 访存带宽 | 占访存上限 | 最大相对误差 |
| --- | --- | --- | --- |
| `x * T.sigmoid(x)`，算在 bf16 上 | 3.44 TB/s | 84% | 0.00961 |
| 整式提到 fp32 | **3.88 TB/s**{ .win } | **95%**{ .win } | **0.00389**{ .win } |

快 13%，误差降到四成。

### 取舍

**fp16 必须提到 fp32**，否则编译不过。**bf16 也应该提**，理由是上面这两栏。

代价是每个元素多两次显式转换（读入与写出各一次），但它们替换掉的是每次运算两侧的隐式转换，总数更少。

### 代码

反例，算在存储 dtype 上：

```python
@staticmethod
def op_func(x):
    return x * T.sigmoid(x)      # bf16 上能编译但更慢；fp16 上编译不过
```

正例，整式提到 fp32：

```python
@staticmethod
def op_func(x):
    one = T.cast(1.0, "float32")
    wide = T.cast(x, "float32")
    return wide / (one + T.exp(-wide))
```

正例里用除法而不是「乘以 sigmoid」，是为了少一次运算 —— 但**不要把它当成省了一条指令**。数过 SASS：fp32 的除法被展开成 Newton-Raphson 序列，`w / (1 + exp(-w))` 是 22 条 FFMA、4 条 MUFU，而 `w * (1 / (1 + exp(-w)))` 是 14 条 FFMA、6 条 MUFU。除法是用 FFMA 换 MUFU，不是更少的指令。

## 2. 用恒等式减少超越函数的个数 {#identities}

一个式子里出现多个超越函数时，先看有没有代数等价的写法能减少个数。以 mish 为例，直接照定义写要 exp、log、tanh 三个；设 $e = \exp(x)$，有两种等价形式只需要一个 exp：

- $\tanh(\log(1+e)) = \dfrac{s^2 - 1}{s^2 + 1}$，其中 $s = 1 + e$ —— $e$ 小于 1 的 eps 时 $s^2 - 1$ 完全相消，不能用。
- $\tanh(\log(1+e)) = \dfrac{e^2 + 2e}{e^2 + 2e + 2}$ —— 没有减法，可以用。

**代数等价的形式里要挑没有相消误差的那个。** 两式都对，但第一式在 $e$ 很小时把两个几乎相等的数相减。

### 溢出用分支处理，不用更高精度

第二式在 $x$ 大时会溢出：$e^2 + 2e$ 越过 fp32 的 $3.4 \times 10^{38}$。实测不带分支时，**首个 NaN 出现在 $x = 42.5$**。

而在这个区间里那个比值到 fp32 每一位都是 1，所以取一个分支直接返回 $x$ 即可。阈值取 20 是保守的，离 42.5 还有很大余量；加上分支后 $x \in [10, 60]$ 全区间无 NaN 或 Inf，最大相对误差 $1.17 \times 10^{-7}$。

### 实测

H200，SM 时钟锁在 1830 MHz。带宽用 fp16 `mish`，输入 512 MB（大于 60 MiB 的 L2），读加写共 1 GB。精度另测：fp32 输入输出，200 万个探测点覆盖 $x \in [-50, 50]$，与 float64 参考值比：

| 写法 | 访存带宽 | 占访存上限 | fp32 上的最大相对误差 |
| --- | --- | --- | --- |
| $x \cdot \tanh(\log(1+e))$，三个超越函数 | 2.01 TB/s | 49% | **1**（在 $x = -50$，全损） |
| 恒等式加饱和分支，一个超越函数 | **3.46 TB/s**{ .win } | **85%**{ .win } | **$2.9 \times 10^{-7}$**{ .win }（在 $x = -4.8$） |

**快 72%，而且直接形式在极负 $x$ 上是完全错的。** 机制是 $x$ 很负时 $e = \exp(x)$ 极小，$1 + e$ 在 fp32 里舍成 1，$\log(1) = 0$，整个结果变成 0 —— 相对误差 1 就是这么来的。恒等式那一支没有这一步，所以不受影响。

### 代码

反例，三个超越函数：

```python
@staticmethod
def op_func(x):
    one = T.cast(1.0, "float32")
    return x * T.tanh(T.log(one + T.exp(x)))
```

正例，一个超越函数加一个分支：

```python
_MISH_SATURATION = 20.0

@staticmethod
def op_func(x):
    two = T.cast(2.0, "float32")
    wide = T.cast(x, "float32")
    e = T.exp(wide)
    saturated = e * e + two * e
    return T.if_then_else(
        wide > T.cast(_MISH_SATURATION, "float32"),
        wide,
        wide * saturated / (saturated + two),
    )
```
