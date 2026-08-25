# 超越函数的开销与精度

这一页记录 elementwise 算子里超越函数带来的开销与精度问题。它们不属于访存受限 —— 恰恰相反，只有当算术开始成为瓶颈时才需要考虑这些。

这一页的数字都是**访存带宽** —— 搬运的字节数（读一次加写一次）除以耗时，单位 TB/s。elementwise 算子本该是访存受限的，所以这个数离机器的访存上限（这块卡实测 4.07 TB/s）有多远，衡量的就是算术占掉了多少时间：越接近上限，说明算术越不再是瓶颈。

超越函数走 MUFU（special function unit），吞吐是 fp32 FMA 的 1/8 —— compute capability 9.0 上每 SM 每周期 16 个结果（对比 FMA 这个数字是 128）。一个 elementwise 算子里若含两三个精确超越函数，算术的耗时便不再可以忽略。

## 1. 在 fp32 上做超越函数与除法 {#fp32-transcendental}

**应该注意**：`T.exp`、`T.log`、除法直接作用在 fp16 或 bf16 的值上，既慢又不准。

超越函数不走普通的浮点流水线，而是走 MUFU（special function unit），而 MUFU 只有 fp32 的形式。窄类型的超越函数无论怎么写都要转到 fp32 再转回：写在存储 dtype 上，这对转换落在**每一次运算的两侧**；写在 fp32 上，整个式子只在读入与写出各转一次。

中间量的精度也不同：留在 fp16 意味着每一步都被舍到 11 位有效位。

反例

```python
@staticmethod
def op_func(x):
    return x * T.sigmoid(x)      # x 是存储 dtype，指数就跑在那里
```

正例

```python
@staticmethod
def op_func(x):
    one = T.cast(1.0, "float32")
    wide = T.cast(x, "float32")
    return wide / (one + T.exp(-wide))   # 顺带把乘法折进除法，少一次乘法
```

实测

H200，256M 个元素的 `silu`，访存带宽：fp16 3.54 → 4.05 TB/s（占访存上限 87% → 99%），bf16 3.74 → 4.04 TB/s。对 fp32 参考值的最大误差同时减半。

## 2. 用恒等式减少超越函数的个数 {#identities}

**应该注意**：超越函数走 MUFU，吞吐是 fp32 FMA 的 1/8；精确版 `expf`、`logf` 在 MUFU 之外还要做区间归约 —— 读生成的代码可以数出来是十几条指令的序列，三个叠起来量级上相当于上百个 FMA。这两个数字是数出来的估计，不是手册给的吞吐。

代数等价的形式里，要选没有相消误差的那个。以 mish 为例，设 $e = \exp(x)$：

- $\tanh(\log(1+e)) = \dfrac{s^2 - 1}{s^2 + 1}$，其中 $s = 1 + e$ —— $e$ 小于 1 的 eps 时 $s^2 - 1$ 完全相消。
- $\tanh(\log(1+e)) = \dfrac{e^2 + 2e}{e^2 + 2e + 2}$ —— 没有减法。

溢出边界靠分支处理，不靠更高精度：$x$ 大时 $e^2$ 会越过 fp32 的 $3.4 \times 10^{38}$，而同一区间里那个比值到 fp32 每一位都是 1，取分支返回 $x$ 同时解决精度与溢出。

反例

```python
@staticmethod
def op_func(x):
    one = T.cast(1.0, "float32")
    return x * T.tanh(T.log(one + T.exp(x)))   # exp、log、tanh 三个
```

正例

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

实测

H200，26.2M 个 fp16 元素的 `mish`，访存带宽 1.85 → 3.36 TB/s，即访存上限的 45% → 83% —— 改写之后算术不再是瓶颈。同时更准：原式在极负 $x$ 上 $\log(1+e)$ 把小量舍成零，fp32 全域最大相对误差 1.0，新形式 3.3e-07。
