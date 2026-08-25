# 访存受限 kernel 的调优

## 什么是访存受限

在 GPU 上，一个 kernel 跑多快，取决于算力与带宽哪一个先成为瓶颈。TileOPs 用 [macro benchmark](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/hardware) 测算出一个**[校准系数](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/perf/profiles/h200.yaml)**：硬件 spec 给出的理论峰值乘以它得到实际可达的有效值，以此作为性能优化的指导标准。我们在H200 上实测出 [fp32 FMA 的算力为 **57.27** TFLOP/s，访存带宽为 **4.07** TB/s](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/perf/profiles/h200.yaml)；两者相除，得到的就是 roofline 的**拐点**（ridge point）：

<figure class="roofline" markdown="1">

<svg class="tf-roofline" viewBox="0 0 520 306" role="img" aria-label="H200 的 roofline：带宽上限 4.07 TB/s 的斜线在每字节 14 次浮点运算处与 57.27 TFLOP/s 的算力上限相交；silu 位于斜线左端，只够到算力上限的 5%。">
<path class="tf-rl-region" d="M 52.0 250.0 L 52.0 217.9 L 357.4 67.1 L 357.4 250.0 Z"/>
<line class="tf-rl-grid" x1="115.4" y1="52" x2="115.4" y2="250"/>
<text class="tf-rl-tick" x="115.4" y="270" text-anchor="middle">1</text>
<line class="tf-rl-grid" x1="178.9" y1="52" x2="178.9" y2="250"/>
<text class="tf-rl-tick" x="178.9" y="270" text-anchor="middle">2</text>
<line class="tf-rl-grid" x1="242.3" y1="52" x2="242.3" y2="250"/>
<text class="tf-rl-tick" x="242.3" y="270" text-anchor="middle">4</text>
<line class="tf-rl-grid" x1="305.7" y1="52" x2="305.7" y2="250"/>
<text class="tf-rl-tick" x="305.7" y="270" text-anchor="middle">8</text>
<line class="tf-rl-grid" x1="369.1" y1="52" x2="369.1" y2="250"/>
<text class="tf-rl-tick" x="369.1" y="270" text-anchor="middle">16</text>
<line class="tf-rl-grid" x1="432.6" y1="52" x2="432.6" y2="250"/>
<text class="tf-rl-tick" x="432.6" y="270" text-anchor="middle">32</text>
<line class="tf-rl-grid" x1="496.0" y1="52" x2="496.0" y2="250"/>
<text class="tf-rl-tick" x="496.0" y="270" text-anchor="middle">64</text>
<line class="tf-rl-grid" x1="52" y1="250.0" x2="496" y2="250.0"/>
<text class="tf-rl-tick" x="44" y="255.0" text-anchor="end">1</text>
<line class="tf-rl-grid" x1="52" y1="218.7" x2="496" y2="218.7"/>
<text class="tf-rl-tick" x="44" y="223.7" text-anchor="end">2</text>
<line class="tf-rl-grid" x1="52" y1="187.4" x2="496" y2="187.4"/>
<text class="tf-rl-tick" x="44" y="192.4" text-anchor="end">4</text>
<line class="tf-rl-grid" x1="52" y1="156.0" x2="496" y2="156.0"/>
<text class="tf-rl-tick" x="44" y="161.0" text-anchor="end">8</text>
<line class="tf-rl-grid" x1="52" y1="124.7" x2="496" y2="124.7"/>
<text class="tf-rl-tick" x="44" y="129.7" text-anchor="end">16</text>
<line class="tf-rl-grid" x1="52" y1="93.4" x2="496" y2="93.4"/>
<text class="tf-rl-tick" x="44" y="98.4" text-anchor="end">32</text>
<line class="tf-rl-grid" x1="52" y1="62.1" x2="496" y2="62.1"/>
<text class="tf-rl-tick" x="44" y="67.1" text-anchor="end">64</text>
<line class="tf-rl-axis" x1="52" y1="250" x2="496" y2="250"/>
<line class="tf-rl-axis" x1="52" y1="52" x2="52" y2="250"/>
<text class="tf-rl-axis-title" x="4" y="34">可达算力 TFLOP/s（对数轴）</text>
<text class="tf-rl-axis-title" x="496" y="294" text-anchor="end">算术强度：每字节的浮点运算数（对数轴）</text>
<polyline class="tf-rl-roof" points="52.0,217.9 357.4,67.1 496.0,67.1"/>
<line class="tf-rl-drop" x1="357.4" y1="67.1" x2="357.4" y2="250"/>
<circle class="tf-rl-ridge" cx="357.4" cy="67.1" r="6"/>
<text class="tf-rl-label tf-rl-label--ridge" x="370.4" y="89.1">拐点</text>
<text class="tf-rl-sub" x="370.4" y="107.1">每字节 14 次</text>
<circle class="tf-rl-point" cx="89.1" cy="199.6" r="5.5"/>
<text class="tf-rl-label tf-rl-label--point" x="102.1" y="219.6">silu (fp16)</text>
<text class="tf-rl-sub" x="102.1" y="237.6">3 次 / 4 字节，只够到 3.1</text>
<text class="tf-rl-roof-label" x="152.5" y="153.3" transform="rotate(-30 152.5 153.3)">带宽上限 4.07 TB/s</text>
<text class="tf-rl-roof-label" x="492.0" y="54.1" text-anchor="end">算力上限 57.3 TFLOP/s</text>
</svg>

<figcaption>图上这条折线就是 roofline，任何 kernel 的性能点都落在它以下。拐点以左，上限是「算术强度 × 带宽」，随强度线性上升；拐点以右，上限就是算力峰值，不再随强度变化。<code>silu</code> 每搬运 4 个字节只做 3 次运算，算术强度只有拐点的 1/19，所以即便带宽完全用满，也只能达到算力上限的 5%。</figcaption>

</figure>

[Elementwise](https://tile-ai.github.io/TileOPs.github.io/api/elementwise/) 与 [Reduction](https://tile-ai.github.io/TileOPs.github.io/api/reduction/) 是典型的访存受限 kernel。在调优这一类kernel的性能时，我们反复遇到过几类常见问题，这里把他们整理成条目，每条给出：什么情况下触发、原因分成、反例与正例的代码。在进入下文之前，我们先给出两条会被反复用到的硬件事实：

- **一次 global memory 读取的数据访问单位是 32 字节的 sector。** sector 同时也是缓存内部搬运数据的单位：

    | 名称 | 大小 | 是什么 |
    | --- | --- | --- |
    | cache line | 128 字节 | L1 与 L2 的 cache line，也是 tag（查找时用的键）的单位 |
    | **sector** | **32 字节** | 一条 cache line 由 4 个 sector 组成，L1 与 L2 之间按 sector 传输 |

    缓存查找时以 cache line 为单位，搬运数据时以 sector 为单位：某个 sector 未命中，L1 就只向 L2 请求这一个 sector，不必把整条 cache line 都拉过来。由此得到的结论是**取 1 个字节和取满 32 个字节的代价相同**。于是一条访存指令的好坏，看的是 `真正用到的字节 / (覆盖的 sector 数 × 32)`。


- **shared memory 由 32 条 bank 构成，每条 bank 宽 4 字节，32 条可以同时被访问。** 一个地址落在哪条 bank 上，由 `(字节地址 / 4) mod 32` 决定。多个线程并发访问 shared memory 时，只要它们落在不同的 bank 上，这些请求就能被并行地服务完。

    落在同一条 bank 上则不然。同一个 warp 内若有多个线程访问一条 bank 上的**不同** word，硬件会把这次请求拆成若干次无冲突的请求依次完成，拆分的次数就是冲突的路数。例外是**读取同一个** word：任意两个线程只要落在同一个 word 内（哪怕取的是其中不同的字节），彼此就不冲突，这个 word 会被广播给所有请求它的线程；分布在不同 bank 上的多个广播还会合并为一次 multicast。这两种情形都不产生冲突。（若多个线程写入同一地址，只有一个写入生效，是哪一个未定义。）

## 1. 一个线程要消费多个元素时，把搬运和消费分开 {#rule-1}

一个线程要连续消费 $V$ 个元素时，最直觉的写法是让它自己去 global memory 取那一段 —— `x[base + tx * V + c]`，第 `tx` 号线程拿第 `tx` 段，段内用 `c` 取偏移。这种切分叫 blocked，它的问题是每条指令覆盖的字节远多于用到的。解法是先用 `T.copy` 或 `T.Parallel` 把这一段搬进 shared memory 或 fragment，再从缓存上按消费需要的顺序读。**把 global memory 的读法改成 striped（相邻线程取相邻元素、各自跳着走）不是解法** —— 见下面的实测。

blocked 写法里，固定 `c` 时相邻线程的地址相差 V 个元素。V = 8 的 fp16 就是 16 字节，一个 warp 的 32 个线程覆盖 512 字节即 16 个 sector；每个 sector 里落进相邻两个线程的元素，用到 4 字节，效率 4/32 = 12.5%。

但这份浪费不落在 DRAM 上：相邻迭代命中的是同一批 sector，它们仍在 L1 里。代价出在 L1 的 sector 请求数与 LSU 的发射数上。这解释了两件事 —— 实测差距是 10% 左右而不是十几倍，以及**单纯把读法换成 striped 并不会更快**，因为那只是换了一种同样窄的访问。

真正省下的是指令：搬运交给 `T.copy` 或 `T.Parallel`，它们的线程映射由布局推导给出，一次搬 16 字节；消费再从片上按任意顺序读，片上没有 sector 的概念。串行前缀（scan）这类要求一个线程持有连续一段的计算，也只有这样才能同时满足合并与顺序两个要求。

反例

```python
# 每个线程直接从 global memory 取自己那一段连续的 V 个元素
for c in T.serial(V):
    acc[0] = acc[0] * T.cast(x[row, (t * threads + tx) * V + c], "float32")
```

正例

```python
staged = T.alloc_shared((chunk,), dtype)

# 搬运：线程映射由 T.Parallel 给出，一次 16 字节
for i in T.Parallel(chunk):
    staged[i] = x[row, t * chunk + i]
T.sync_threads()

# 消费：片上，顺序随计算需要
for c in T.serial(V):
    acc[0] = acc[0] * T.cast(staged[tx * V + c], "float32")
```

实测

H200，2048×4096 bf16 的按行 `prod`，三种写法：

| 写法 | 带宽 |
| --- | --- |
| blocked，直接读 global memory | 2.15 TB/s |
| striped，直接读 global memory | 2.03 TB/s |
| 经 shared 中转 | **2.34 TB/s** |

同一形状的按行 `cumsum`（fp16）：不用 shared memory、每线程从 global memory 按 blocked 读进寄存器 0.4 到 1.1 TB/s，经 shared 中转 2.77 TB/s。

## 2. 每线程按固定 stride 读 shared memory 时，给 stride 加 pad {#rule-2}

**应该注意**：[第 1 条](#rule-1)的 `staged[tx, j]` 里，每个线程的起始地址相差一整个 chunk。这个 stride 换算成 bank word 后如果是 32 的倍数，整个 warp 落在同一个 bank 上。

bank 编号 = (字节地址 / 4) mod 32。fp16、每线程 chunk 为 64 列时，stride 是 128 字节即 32 个 word，`32 · tx mod 32 = 0` —— 一个 warp 的 32 个线程**全部落在 bank 0，32 路串行**。

pad 8 个 fp16 之后 stride 是 144 字节即 36 个 word，`gcd(36, 32) = 4`，落在 8 个不同 bank 上，退化到 4 路。

pad 取 16 字节的倍数，每个 chunk 才仍然 16 字节对齐；否则 SASS 里的 `LDS.128` 与 `STS.128` 失去对齐，搬入一侧反而丢掉向量化。16 字节是 PTX 一次向量访问的最大宽度，也要求按 16 字节对齐。

**pad 之后要重扫最优 chunk。** 冲突消掉后，更大的 chunk 意味着更少线程争用同一批 bank，最优点会往大移。

反例

```python
staged = T.alloc_shared((threads, chunk), dtype)   # stride = chunk
```

正例

```python
_PAD_BYTES = 16
pad = _PAD_BYTES // torch.empty(0, dtype=getattr(torch, dtype)).element_size()
staged = T.alloc_shared((threads, chunk + pad), dtype)   # stride = chunk + pad
```

实测

H200，2048×4096 fp16 的按行 `cumsum`：chunk = 64 不加 pad 时 1.31 TB/s，加 16 字节 pad 后 3.44 TB/s。同时最优 chunk 从 16 列移到 64 列 —— 这就是上面说的「要重扫」。

`tilelang.layout.make_swizzled_layout` 是另一条路，不需要 pad，但对形状敏感：同一个 kernel 在 chunk = 128 时降到 1.13 TB/s，而 pad 是 3.30。

## 3. 整行装得进寄存器时，直接读进 fragment {#rule-3}

**应该注意**：一行能被一个 block 放进寄存器时，一次 `T.copy` 读进 fragment 比经 shared 中转少一趟往返；元素的列下标就是遍历 fragment 的循环变量，不需要另外维护。

fragment 在寄存器里。`T.copy` 的线程映射由布局推导给出，本身就是 striped 的，所以这条与[第 1 条](#rule-1)一致。

装得下与否由每线程 255 个寄存器决定：每线程持有的 32 位量等于 `元素数 × 元素字节 / 4`。行宽固定时这个量只随线程数变 —— 4096 列 fp16 在 256 线程下是 8 个寄存器，在 64 线程下是 32 个。**所以线程数要随行宽定，而不是取一个固定值。**

超过 255，编译器把多出来的量放进 local memory。local memory 物理上就在 global memory 里，延迟与带宽随之；虽然默认经 L1 与 L2 缓存，一次溢出仍然把寄存器访问换成了访存。

反例

```python
# 每线程一次一个元素地走这一行，列下标自己算
for it in T.serial(iterations):
    index = it * threads + tx
    if index < N:
        update(best, key_of(x[row, index]), index)
```

正例

```python
row_frag = T.alloc_fragment((1, N), dtype)
T.copy(x[row : row + 1, :], row_frag)

# 列下标就是循环变量
for _, column in T.Parallel(1, N):
    update(best, key_of(row_frag[0, column]), column)
```

实测

H200，2048×4096 fp16 的按行 `argmax`：逐元素走 global memory 2.02 TB/s，读进 fragment 2.21 TB/s。行越宽差距越大 —— 256×32768 上是 1.65 对 2.12 TB/s。

## 4. 存储 dtype 窄于 fp32 时，把超越函数和除法提到 fp32 {#rule-4}

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

H200，256M 个元素的 `silu`：fp16 3.54 → 4.05 TB/s，bf16 3.74 → 4.04 TB/s；对 fp32 参考值的最大误差同时减半。

## 5. 一个式子里出现多个超越函数时，先找恒等式 {#rule-5}

**应该注意**：超越函数走 MUFU，吞吐是 fp32 FMA 的 1/8；精确版 `expf`、`logf` 在 MUFU 之外还要做区间归约 —— 读生成的代码可以数出来是十几条指令的序列，三个叠起来量级上相当于上百个 FMA。这两个数字是数出来的估计，不是手册给的吞吐。

代数等价的形式里，要选没有相消误差的那个。以 mish 为例，设 `e = exp(x)`：

- `tanh(log(1 + e)) = (s² − 1) / (s² + 1)`，其中 `s = 1 + e` —— `e` 小于 1 的 eps 时 `s² − 1` 完全相消。
- `tanh(log(1 + e)) = (e² + 2e) / (e² + 2e + 2)` —— 没有减法。

溢出边界靠分支处理，不靠更高精度：`x` 大时 `e²` 会越过 fp32 的 3.4e38，而同一区间里那个比值到 fp32 每一位都是 1，取分支返回 `x` 同时解决精度与溢出。

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

H200，26.2M 个 fp16 元素的 `mish`：1.85 → 3.36 TB/s。同时更准 —— 原式在极负 `x` 上 `log(1 + e)` 把小量舍成零，fp32 全域最大相对误差 1.0，新形式 3.3e-07。

## 6. 输出比输入窄时，每线程多取一次输入 {#rule-6}

**应该注意**：每线程元素数按输入宽度凑一次 128 位访问时，更窄的输出得到的存储指令覆盖不了同样的字节数。

每线程 16 字节的来由是 16 字节为 PTX 一次向量访问的最大宽度（SASS 里是 `LDG.E.128`），32 × 16 = 512 字节正好 4 条 128 字节 cache line。但这个尺寸按输入算：fp32 输入取 4 个元素时，1 字节的输出每线程只有 4 字节，一条存储指令覆盖 32 × 4 = 128 字节即 4 个 sector，而对应的读覆盖 512 字节即 16 个 sector —— **存储侧每搬一个字节付 4 倍的指令数**。

每线程元素数加倍后，读变成两条指令、存储变成每线程 8 字节，总指令数下降。存储本来就完全合并，所以这是指令数问题，不是 DRAM 效率问题。

反例

```python
npt = _BYTES_PER_THREAD // elem_bytes        # 只看输入宽度
```

正例

```python
npt = _BYTES_PER_THREAD // elem_bytes
if stored_bytes < elem_bytes:
    npt *= 2                                 # 结果更窄，取两次输入访问
```

实测

H200，16M 个 fp32 元素的 `isnan`（bool 输出）：3.34 → 3.63 TB/s；256M 时 4.10 → 4.36 TB/s。两个输入的比较类算子在同一区间内是平的，因为它们的线程本来就多持有一份输入。

## 7. kernel 产生 bool 时，用 int8 存储 {#rule-7}

**应该注意**：向量化的 bool 在 CUDA codegen 里没有对应类型，会把每线程元素数卡在 4。

CUDA 没有 8 个 bool 的向量类型，codegen 发不出对应的 128 位存储。把一个 bool 结果写进每线程 8 个元素的 fragment，编译当场报错：

```
Fatal: Cannot convert type boolx8 to CUDA type
```

int8 有 `char4` 这类向量类型。`torch.bool` 每个元素占 1 字节，而正例里的 `T.if_then_else` 已经把结果规范成 0 或 1 —— 两种 dtype 的字节模式因此逐字节相同，`.view(torch.bool)` 不拷贝数据。（前提就是这个规范化；直接把任意非零字节当 bool 看是不安全的。）

再进一步：谓词由 `T.if_then_else` 消费之后，IR 里根本不存在可被加宽的 bool 值 —— 每线程元素数因此不再受这条限制。

反例

```python
def op_func(x):
    return x != x          # 结果是 bool，向量化路径走不通
```

正例

```python
_BOOL_STORAGE_DTYPE = "int8"

def wrapped(x):
    return T.if_then_else(
        op_func(x),
        T.cast(1, _BOOL_STORAGE_DTYPE),
        T.cast(0, _BOOL_STORAGE_DTYPE),
    )

# forward 里把 int8 结果交回为 bool，不拷贝
return result.view(torch.bool)
```

实测

H200，16M 个元素，对同一算子最快的其他实现的比值：`isnan` 从 0.738 到 0.924，`logical_not` 从 0.811 到 0.921。改法抬升的是整个谓词族，不只这两个。

## 8. 传给 T.macro 的表达式先绑到局部变量 {#rule-8}

**应该注意**：`T.macro` 是文本替换，不是函数调用 —— 参数在宏体里出现几次就展开几次。

一个在宏体里被提到四次的参数，如果传进去的是一个调用，那次计算就在生成的 CUDA 里出现四份。这不是编译器一定会消除的公共子表达式：`T.reinterpret` 这类位级操作 nvcc 不一定认作可复用。

判断方法是读生成的代码：`kernel.get_kernel_source()`。

反例

```python
@T.macro
def update(keys, indices, slot, candidate_key, candidate_index):
    if candidate_key > keys[slot] or (
        candidate_key == keys[slot] and candidate_index < indices[slot]
    ):
        keys[slot] = candidate_key
        indices[slot] = T.cast(candidate_index, "int32")

# candidate_key 在宏体里出现四次，于是 key_of 被算四遍
update(keys, indices, slot, key_of(x[row, index]), index)
```

正例

```python
candidate = T.alloc_local((1,), "int32")

candidate[0] = key_of(x[row, index])
update(keys, indices, slot, candidate[0], index)
```

实测

argreduce 的排序键在生成的 CUDA 里由六份减为一份。

## 9. 需要让 -0.0 与 +0.0 落到同一个值时，不要依赖 x + 0.0 {#rule-9}

**应该注意**：IEEE 754 规定 `-0.0 + 0.0 = +0.0`，写法本身没错，但这次加法到不了运行时 —— TileLang 生成的 CUDA 里它已经不见了。

折叠发生在 TileLang（TVM 的算术简化器）这一层：它把「加零」当作恒等变换消除，而这个恒等式对 `-0.0` 恰好不成立。实测同一段代码交给 `nvcc -O3 -arch=sm_90` 时 SASS 里 `FADD` 仍在，所以这不是 nvcc 的行为，指望换编译选项绕开是没用的。**依赖 IEEE 特例做规范化，就要选不会被当作恒等式的运算** —— 掩符号位是位操作，没有代数恒等式可套。

按位比较的排序键尤其要注意：`-0.0` 与 `+0.0` 作为浮点数相等，而位模式不同。两者不同键，`argmax` 在一行 `±0.0` 上就会返回错的下标。

反例

```python
shifted = wide + T.cast(0.0, "float32")      # 简化器直接消掉，-0.0 仍是 -0.0
bits = T.reinterpret(shifted, "int32")
```

正例

```python
bits = T.reinterpret(wide, "int32")
sign = bits >> 31
magnitude = T.bitwise_and(bits, T.int32(0x7FFFFFFF))
ordered = T.bitwise_xor(magnitude, sign) - sign   # 两个零的幅值都是 0，取负仍是 0
```

这个变换对所有非 NaN 的浮点数给出与浮点比较一致的整数序，但它**不是完整的 IEEE `totalOrder`** —— NaN 要单独分出一支处理（`argreduce.py` 里就是 `T.if_then_else(oriented != oriented, ...)` 那一句）。

实测

不是性能项，是正确性项：`argmax` 在一行交替的 `±0.0` 上从返回第一个 `+0.0` 的下标改为返回 0，与 PyTorch 一致。