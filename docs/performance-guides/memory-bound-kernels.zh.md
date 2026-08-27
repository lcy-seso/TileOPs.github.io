# 优化 access pattern

## 什么是访存受限

在 GPU 上，一个 kernel 跑多快，取决于算力与带宽哪一个先成为瓶颈。TileOPs 用 [macro benchmark](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/hardware) 测算出一个**[校准系数](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/perf/profiles/h200.yaml)**：硬件 spec 给出的理论峰值乘以校准系数，得到实际可达的有效值，以此作为性能优化的指导标准。我们在H200 上实测出 [fp32 FMA 的算力为 **57.27** TFLOP/s，访存带宽为 **4.07** TB/s](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/perf/profiles/h200.yaml)。roofline 的**拐点**（ridge point）是带宽斜线与算力上限这两段的交点，两者相除给出它的横坐标，也就是拐点处的算术强度 **14.07 flop/byte** —— 算力与带宽同时用满时，每搬运一个字节对应的浮点运算次数：

<figure class="roofline" markdown="1">

<svg class="tf-roofline" viewBox="0 0 520 306" role="img" aria-label="H200 的 roofline：带宽上限 4.07 TB/s 的斜线在每字节 14 次浮点运算处与 57.27 TFLOP/s 的算力上限相交；silu 位于斜线左端，可达算力为算力上限的 9%。">
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
<circle class="tf-rl-point" cx="135.8" cy="176.5" r="5.5"/>
<text class="tf-rl-label tf-rl-label--point" x="148.8" y="196.5">silu (fp16)</text>
<text class="tf-rl-sub" x="148.8" y="214.5">5 次 / 4 字节，上限 5.1</text>
<text class="tf-rl-roof-label" x="152.5" y="153.3" transform="rotate(-30 152.5 153.3)">带宽上限 4.07 TB/s</text>
<text class="tf-rl-roof-label" x="492.0" y="54.1" text-anchor="end">算力上限 57.3 TFLOP/s</text>
</svg>

<figcaption>图上这条折线就是 roofline，任何 kernel 的性能点都落在它以下。拐点以左，上限是「算术强度 × 带宽」，可达算力随算术强度线性上升；拐点以右，上限就是算力峰值，不再随算术强度变化。<code>silu</code> 每搬运 4 个字节做 5 次运算，算术强度 1.25 flop/byte，只有拐点的 1/11，所以即便带宽完全用满，也只能达到算力上限的 9%。</figcaption>

</figure>

## 确认 DRAM 带宽是否为当前的瓶颈 {#regime}

[Elementwise](https://tile-ai.github.io/TileOPs.github.io/api/elementwise/) 与 [Reduction](https://tile-ai.github.io/TileOPs.github.io/api/reduction/) 是典型的访存受限 kernel。我们把调优这两类 kernel 过程中遇到的常见问题整理成经验法则，每条给出触发的条件、成因，反例以及正例代码。

这些法则都在同一组条件下测得：**输入大于 L2 的 60 MiB，且 block 数足以填满整卡**（H200 有 132 个 SM）。此时 DRAM 带宽是主要瓶颈，访存模式的差别直接反映在性能上。

!!! warning "适用范围"

    这组条件之外，主要性能瓶颈可能由别的因素决定，这里给出的一些结论会反转。

下表按这两个条件划出三个区间，逐行给出判据与本文各条经验法则在其中的用法。表里的瓶颈是主导因素，kernel 越复杂，同时起作用的因素越多：

| 区间 | 判据 | 主要瓶颈 | 各条法则的用法 |
| --- | --- | --- | --- |
| 带宽饱和 | 输入 > 60 MiB，block 数在 SM 数的两倍以上 | DRAM 带宽，即 sector 利用率 | 直接适用 |
| 数据小 | 输入装得进 L2，单次耗时在几十微秒以内 | kernel 发射的固定开销、缓存状态 | 只需避开各条的反例，换 access pattern 没有收益 |
| block 少 | block 数不到 SM 数的两倍 | 每条载入指令的宽度、在飞的字节数 | 保住载入宽度优先，改动逐个实测 |

- **数据小时，发射开销与缓存状态占主导。** 同一个行求和 kernel 的四种 access pattern（fp16，256 线程，时钟未锁）在 65536 × 4096（512 MB）上测得的访存带宽是 4.20 到 4.43 TB/s，彼此相差不超过 6%；换成 2048 × 4096（16 MB，装得进 L2）后单次耗时十几微秒，同一个 access pattern 两次测量之间可差三倍。这个区间里换 access pattern 没有收益，制约性能的是别的因素。
- **block 少时，载入宽度带来的收益大于合并规则算出的差别。** warp 数量不足，靠并发的请求数掩盖访存延迟不再可行，只能让每个请求更宽、每个线程持有更多在飞的字节。以载入宽度换取其他好处的改法，在这个区间都可能反转。

## 1. 合并 global memory 的访存 {#coalescing}

### 背景

**一次 global memory 读取的数据访问单位是 32 字节的 sector。**

| 名称 | 大小 | 是什么 |
| --- | --- | --- |
| cache line | 128 字节 | L1 与 L2 的缓存行，也是缓存查找的单位 |
| **sector** | **32 字节** | 一条 cache line 由 4 个 sector 组成，L1 与 L2 之间按 sector 传输 |

缓存查找时以 cache line 为单位，搬运数据时以 sector 为单位：某个 sector 未命中，L1 就只向 L2 请求这一个 sector，不必把整条 cache line 都拉过来。由此得到的结论是**取 1 个字节和取满 32 个字节的代价相同**。于是一条访存指令的好坏由 **sector 利用率**衡量：`真正用到的字节 / (覆盖的 sector 数 × 32)`。

硬件把一个 warp 的 32 个访问合并成尽可能少的 32 字节事务。事务数最少要同时满足三个因素：

1. **地址连续** —— 同一条指令里 32 个线程的地址首尾相接，不留空洞；
2. **按 32 字节对齐** —— 起始地址是 32 的倍数，一段数据不会多占一个 sector；
3. **每个线程一次取满 16 字节** —— 一条指令覆盖 $32 \times 16 = 512$ 个连续字节，即 16 个满载的 sector。

三个因素同时成立时，这条访存指令对硬件最友好。

一个线程要读 $V$ 个元素时（$V$ = 一行的元素数 / 线程数），有四种 access pattern。

**blocked** —— 每个线程负责一段连续的元素。固定 `c` 时相邻线程的地址相隔 $V$ 个元素，违反第一个因素，sector 利用率是 $1/V$：

```python
for c in T.serial(V):
    acc[0] = acc[0] * X[row, tx * V + c]
```

**striped** —— 相邻线程取相邻元素。地址连续了，但每个线程一次只取一个元素，违反第三个因素，$V$ 个元素要发 $V$ 条指令：

```python
for c in T.serial(V):
    acc[0] = acc[0] * X[row, c * threads + tx]
```

**blocked + 向量化** —— 仍是每线程一段连续的，但改用 `T.vectorized` 一次读满 16 字节，三个因素全部满足：

```python
buf = T.alloc_local((V,), dtype)

for c in T.vectorized(V):
    buf[c] = X[row, tx * V + c]
for c in T.serial(V):
    acc[0] = acc[0] * buf[c]
```

**staged** —— 搬运交给 `T.Parallel`，消费改从 shared memory 读，同样满足三个因素：

```python
sh = T.alloc_shared((threads, V + pad), dtype)

for t, c in T.Parallel(threads, V):
    sh[t, c] = X[row, t * V + c]
T.sync_threads()
for c in T.serial(V):
    acc[0] = acc[0] * sh[tx, c]
```

四种 access pattern 的差别在于：**「哪个线程读哪些元素、一次读多宽」这个映射由谁决定。**

`T.serial` 的语义是循环体由单个线程顺序执行，索引表达式被逐字翻译成访存指令，不做合并也不做向量化 —— 编程者写出的模式就是硬件看到的模式。

`T.vectorized`、`T.Parallel`、`T.copy` 则由 TileLang 的 **layout inference** 决定，三者的差别在于编程者还需要写明多少：`T.vectorized` 要写明每线程一次访问的宽度，线程映射由 layout inference 推导；`T.Parallel` 连宽度也不必写，循环维度怎么分给线程、一次读多宽都由它决定；`T.copy` 只写源和目标两个区域，整段搬运由它生成（需要接管推导结果时，另有 `coalesced_width` 与 `loop_layout` 两个参数）。剩下的向量化、地址对齐、以及在 shared memory 一侧避开 bank 冲突，都由 layout inference 负责 —— 这些正是对硬件友好但手写容易出错的部分。

**用 `T.serial` 手写下标时，上面三个因素要自己逐一保证；交给 layout inference 时，只需写明搬运的范围。** 三者之间怎么选、各自能跑到多少带宽，见下面的实测。

<figure class="access-patterns" markdown="1">

<svg class="tf-access" viewBox="0 0 520 352" role="img" aria-label="三种访存模式下，一个 warp 的一条读取指令触及的元素，以及硬件因此取回的 sector。blocked 覆盖 4 个 sector，每个只用到 2 个元素；向量化后的 blocked 与 striped 都完全合并；staged 的搬运阶段与 striped 一样合并，消费阶段在 shared memory 上，不存在 sector。">
<text class="ap-title" x="0" y="38.0">blocked</text>
<text class="ap-sub" x="0" y="52.0">tx * V + c</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="28.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="229.0" y="28.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="322.0" y="28.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="415.0" y="28.0" width="88.0" height="22.0" rx="2"/>
<text class="ap-owner" x="141.5" y="23.0" text-anchor="middle">0</text>
<text class="ap-owner" x="185.5" y="23.0" text-anchor="middle">1</text>
<text class="ap-owner" x="234.5" y="23.0" text-anchor="middle">2</text>
<text class="ap-owner" x="278.5" y="23.0" text-anchor="middle">3</text>
<text class="ap-owner" x="327.5" y="23.0" text-anchor="middle">4</text>
<text class="ap-owner" x="371.5" y="23.0" text-anchor="middle">5</text>
<text class="ap-owner" x="420.5" y="23.0" text-anchor="middle">6</text>
<text class="ap-owner" x="464.5" y="23.0" text-anchor="middle">7</text>
<rect class="ap-cell ap-cell--t0" x="136.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="147.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="158.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="169.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="180.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="191.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="202.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="213.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="229.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="240.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="251.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="262.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="273.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="284.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="295.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="306.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="322.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="333.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="344.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="355.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="366.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="377.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="388.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="399.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="415.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="426.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="437.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="448.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="459.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="470.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="481.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="492.0" y="28.0" width="11.0" height="22.0"/>
<text class="ap-note" x="136.0" y="65.0">取回 4 个 sector，每个只用到 2 / 8 个元素</text>
<text class="ap-title" x="0" y="108.0">blocked + 向量化</text>
<text class="ap-sub" x="0" y="122.0">T.vectorized(V)</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="98.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="229.0" y="98.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="322.0" y="98.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="415.0" y="98.0" width="88.0" height="22.0" rx="2"/>
<text class="ap-owner" x="141.5" y="93.0" text-anchor="middle">0</text>
<text class="ap-owner" x="185.5" y="93.0" text-anchor="middle">1</text>
<text class="ap-owner" x="234.5" y="93.0" text-anchor="middle">2</text>
<text class="ap-owner" x="278.5" y="93.0" text-anchor="middle">3</text>
<text class="ap-owner" x="327.5" y="93.0" text-anchor="middle">4</text>
<text class="ap-owner" x="371.5" y="93.0" text-anchor="middle">5</text>
<text class="ap-owner" x="420.5" y="93.0" text-anchor="middle">6</text>
<text class="ap-owner" x="464.5" y="93.0" text-anchor="middle">7</text>
<rect class="ap-cell ap-cell--t0" x="136.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="147.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="158.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="169.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="180.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="191.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="202.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="213.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="229.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="240.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="245.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="251.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="256.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="262.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="267.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="273.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="284.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="289.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="295.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="300.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="306.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="311.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="322.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="333.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="338.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="344.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="349.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="355.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="360.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="366.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="377.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="382.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="388.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="393.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="399.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="404.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="415.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="426.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="431.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="437.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="442.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="448.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="453.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="459.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="470.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="475.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="481.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="486.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="492.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="497.5" cy="109.0" r="3.2"/>
<text class="ap-note" x="136.0" y="135.0">一条 16 字节向量读，取回 4 个 sector，全部用满</text>
<text class="ap-title" x="0" y="182.0">striped</text>
<text class="ap-sub" x="0" y="196.0">c * threads + tx</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="172.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector" x="229.0" y="172.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector" x="322.0" y="172.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector" x="415.0" y="172.0" width="88.0" height="22.0" rx="2"/>
<text class="ap-owner" x="141.5" y="167.0" text-anchor="middle">0</text>
<text class="ap-owner" x="152.5" y="167.0" text-anchor="middle">1</text>
<text class="ap-owner" x="163.5" y="167.0" text-anchor="middle">2</text>
<text class="ap-owner" x="174.5" y="167.0" text-anchor="middle">3</text>
<text class="ap-owner" x="185.5" y="167.0" text-anchor="middle">4</text>
<text class="ap-owner" x="196.5" y="167.0" text-anchor="middle">5</text>
<text class="ap-owner" x="207.5" y="167.0" text-anchor="middle">6</text>
<text class="ap-owner" x="218.5" y="167.0" text-anchor="middle">7</text>
<rect class="ap-cell ap-cell--t0" x="136.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="147.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="158.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="169.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="180.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="191.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="202.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="213.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="229.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="240.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="251.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="262.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="273.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="284.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="295.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="306.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="322.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="333.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="344.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="355.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="366.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="377.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="388.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="399.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="415.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="426.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="437.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="448.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="459.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="470.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="481.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="492.0" y="172.0" width="11.0" height="22.0"/>
<text class="ap-note" x="136.0" y="209.0">取回 1 个 sector，8 / 8 个元素全部用到</text>
<text class="ap-title" x="0" y="256.0">staged</text>
<text class="ap-sub" x="0" y="270.0">T.Parallel 整行搬运</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="246.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="229.0" y="246.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="322.0" y="246.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="415.0" y="246.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-cell" x="136.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="147.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="158.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="169.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="180.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="191.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="202.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="213.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="229.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="240.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="245.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="251.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="256.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="262.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="267.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="273.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="284.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="289.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="295.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="300.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="306.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="311.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="322.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="333.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="338.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="344.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="349.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="355.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="360.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="366.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="377.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="382.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="388.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="393.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="399.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="404.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="415.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="426.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="431.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="437.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="442.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="448.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="453.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="459.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="470.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="475.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="481.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="486.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="492.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="497.5" cy="257.0" r="3.2"/>
<text class="ap-note" x="136.0" y="283.0">取回 4 个 sector，全部用满</text>
<text class="ap-sub" x="0" y="318.0">staged[tx * V + c]</text>
<rect class="ap-sector ap-sector--onchip" x="136.0" y="294.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--onchip" x="229.0" y="294.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--onchip" x="322.0" y="294.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--onchip" x="415.0" y="294.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-cell" x="136.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="147.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="158.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="169.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="180.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="191.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="202.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="213.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="229.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="240.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="245.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="251.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="256.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="262.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="267.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="273.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="284.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="289.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="295.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="300.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="306.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="311.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="322.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="333.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="338.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="344.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="349.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="355.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="360.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="366.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="377.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="382.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="388.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="393.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="399.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="404.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="415.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="426.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="431.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="437.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="442.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="448.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="453.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="459.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="470.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="475.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="481.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="486.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="492.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="497.5" cy="305.0" r="3.2"/>
<text class="ap-note" x="136.0" y="331.0">在 shared memory 上消费，不存在 sector</text>
<text class="ap-scale" x="0" y="348.0">示意图：fp32、8 个线程、V = 4，一个 sector 装 8 个元素。正文实测用 bf16、256 线程、V = 16，形状相同。</text>
</svg>

<figcaption>一条读取指令。紫点是各线程在这条指令里读到的元素，青色底是硬件因此取回的 sector；格子上方是线程号，深浅交替标出线程边界。最后一行的紫色虚线框是 shared memory，那里不按 sector 组织。</figcaption>

</figure>

### 实测

我们对两个 workload 在 H200 上进行实测，比较上面四种 access pattern 各自能跑到多少**访存带宽**（搬运的字节数除以 kernel 耗时，单位 TB/s），这两个 workload 的计算对元素的处理顺序有不同要求 —— 这个要求会决定哪几种 access pattern 可用。

测试中 SM 时钟锁在 1830 MHz；输入 bf16 的 $65536 \times 4096$（512 MB，**必须大于 L2 的 60 MiB**，否则测到的是 L2 带宽）；每个配置跑三次，三次的结果一致到 ±0.5%。staged 在表里占两列：一列不加 pad（此时 stride 恰好是 $V$ 个 word，产生 bank 冲突，见[第 2 条](#bank-conflict)），一列是在若干个 pad 取值中测到的最优值。

**workload 1，按行求乘积**：一行的元素相乘，先后顺序不影响结果，每行只输出一个值，四种 access pattern 都可用。

| 线程数 | $V$ | blocked<br>TB/s | striped<br>TB/s | blocked + 向量化<br>TB/s | staged 不加 pad<br>TB/s | staged 加 pad<br>TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 8 | 3.02 | 3.04 | **3.22**{ .win } | 3.05 | 3.02 |
| 256 | 16 | 1.83 | 3.31 | **3.81**{ .win } | 3.79 | 3.79 |
| 128 | 32 | 0.95 | 3.36 | **3.99**{ .win } | 3.81 | 3.98 |
| 64 | 64 | 0.48 | 3.49 | 3.32 | 2.80 | **4.02**{ .win } |
| 32 | 128 | 0.47 | 3.43 | 3.11 | 2.83 | **3.88**{ .win } |

**workload 2，按行求串行前缀积**：每个位置的结果依赖它左边的全部元素，顺序不能变，整行都要写回。striped 在这里不可用（线程无法持有连续的一段）。

| 线程数 | $V$ | blocked<br>TB/s | blocked + 向量化<br>TB/s | staged 不加 pad<br>TB/s | staged 加 pad<br>TB/s |
| --- | --- | --- | --- | --- | --- |
| 512 | 8 | 0.92 | **4.20**{ .win } | 2.47 | 3.15 |
| 256 | 16 | 0.42 | **3.85**{ .win } | 1.63 | 3.29 |
| 128 | 32 | 0.34 | 2.76 | 0.89 | **3.35**{ .win } |
| 64 | 64 | 0.26 | 1.80 | 0.46 | **3.69**{ .win } |
| 32 | 128 | 0.27 | 1.60 | 0.46 | **3.24**{ .win } |

### 取舍

1. **逐元素的 blocked 在 $V > 1$ 时总是最差的 access pattern。** 固定 `c` 时相邻线程的地址相隔 $V$ 个元素，sector 利用率是 $1/V$，所以 $V$ 越大越差 —— workload 1 的表里从 $V = 8$ 的 3.02 掉到 $V = 64$ 的 0.48。这个关系由访存合并的规则决定，不随形状改变。

2. **$V$ 小时用向量化的 blocked；$V$ 大到寄存器压力压低占用率时，改用加了 pad 的 staged。** 向量化把整段留在寄存器里（bf16 是每线程 $V/2$ 个），staged 把它放进 shared memory，用一次同步换回寄存器。翻转点取决于 kernel 里其余部分还剩多少寄存器预算，不是一个固定的 $V$：上面两个 workload 在同一个行宽下就分别落在 $V = 64$ 与 $V = 32$。**这个翻转点要在自己的 kernel 上测。**

3. **staged 的 shared 缓冲要避开 bank 冲突。** 声明成 `(threads, V)` 时 stride 恰好是 $V$ 个 word，$V$ 为 2 的幂就一定产生冲突；pad 的算法与候选见 [pad 随 `chunk` 变](#pad-per-chunk)。workload 2 的表里，同一个配置不加 pad 是 0.46，加 pad 是 3.69。

4. **striped 完全合并，但每个元素要发一条指令。** 所以它好于逐元素的 blocked、差于向量化的 blocked（$V = 16$ 上 3.31 对 1.83 与 3.81），适合改动量比最后一点带宽更重要的场合。它让线程持有的元素不连续，因此要求线程持有连续一段的计算（例如串行前缀）用不了它。

下面两段是推荐 access pattern 的完整模板，`M`、`N`、`V`、`threads`、`pad`、`dtype` 都是编译期常量。本条开头的四段代码里，逐元素的 blocked 是反例，不要照抄；striped 可用但不是最快的一种（取舍第 4 条）。

**推荐的 access pattern，小 $V$** —— 向量化的 blocked：

```python
@T.prim_func
def main(X: T.Tensor((M, N), dtype), Out: T.Tensor((M, threads), "float32")):
    with T.Kernel(M, threads=threads) as row:
        tx = T.get_thread_binding()
        buf = T.alloc_local((V,), dtype)
        acc = T.alloc_local((1,), "float32")
        acc[0] = T.cast(1.0, "float32")

        for c in T.vectorized(V):                  # 一条 16 字节向量读
            buf[c] = X[row, tx * V + c]

        for c in T.serial(V):                      # 消费，顺序随计算需要
            acc[0] = acc[0] * T.cast(buf[c], "float32")

        Out[row, tx] = acc[0]
```

**推荐的 access pattern，大 $V$** —— 加了 pad 的 staged（翻转点见取舍第 2 条）：

```python
@T.prim_func
def main(X: T.Tensor((M, N), dtype), Out: T.Tensor((M, threads), "float32")):
    with T.Kernel(M, threads=threads) as row:
        tx = T.get_thread_binding()
        sh = T.alloc_shared((threads, V + pad), dtype)   # pad 的选法见第 2 条
        acc = T.alloc_local((1,), "float32")
        acc[0] = T.cast(1.0, "float32")

        for t, c in T.Parallel(threads, V):        # 搬运，映射由 layout inference 给出
            sh[t, c] = X[row, t * V + c]
        T.sync_threads()

        for c in T.serial(V):                      # 消费
            acc[0] = acc[0] * T.cast(sh[tx, c], "float32")

        Out[row, tx] = acc[0]
```

## 2. 通过 pad 消除 shared memory 的 bank conflict {#bank-conflict}

### 背景

shared memory 由 32 个 bank 构成，每 bank 宽 4 个字节。将 shared memory 的地址空间按 4 字节划分成 **word**（下文一律按 word 计数），一个地址落在哪个 bank 上，由 `(字节地址 / 4) mod 32` 决定。一个 bank 每周期只能处理一个 word 的访问，多个线程并发访问 shared memory 时，只要访问落在不同的 bank 上，它们就在同一个周期一起完成。多个线程落在同一条 bank 上时，又分三种情形：

1. **访问的是不同的 word** —— 硬件把这次请求拆成若干次无冲突的请求依次完成，拆分的次数就是**冲突的路数**。
2. **读取的是同一个 word** —— 任意两个线程只要落在同一个 word 内（哪怕取的是其中不同的字节），这个 word 会被广播给所有请求它的线程，不产生冲突。分布在不同 bank 上的多个广播还会合并为一次 multicast。
3. **写入的是同一个地址** —— 只有一个写入生效，是哪一个未定义。

访问 shared memory 的 access pattern 应当尽可能做到无 bank conflict。

### 冲突路数由什么决定

我们考虑一种不失通用性的情况。下面这个一维数组 `sh` 在 shared memory 上，每个线程读取其中连续的 `chunk` 个元素：

```python
sh = T.alloc_shared((threads * chunk,), dtype)   # 一维数组，threads * chunk 个元素

for c in T.serial(chunk):
    acc[0] = acc[0] * sh[tx * chunk + c]         # 线程 tx 读自己那一段
```

一个 warp 的 32 个线程同步执行这个循环，同一次迭代里 `c` 对它们取同一个值、`tx` 取 0 到 31：这时 32 个线程以等间隔访问 shared memory。例如 `chunk = 64` 时，这 32 个线程在同一次迭代里读的是第 `c`、第 `64 + c`、第 `128 + c`、…… 第 `1984 + c` 个元素。相邻两个线程相差 `chunk` 个元素，这个差值称为这段访问的 **stride**，记作 $S$，换算成 word 是 $S = \text{chunk} \times E / 4$ 个（$E$ 是元素的字节数）。stride 与数组声明成几维、下标怎么写都无关。

另一个变量是访存指令的位宽。以向量化的方式一次访问 $w$ 个连续的 word，$w$ 取 1、2、4，对应 32 bit、`float2` 的 8 字节、`float4` 的 16 字节。于是线程 $t$ 读的是第 $St$ 到 $St + w - 1$ 个 word。

两条前提成立时，冲突路数可以直接从上面的硬件事实数出来：

1. **$S$ 是整数个 word。** fp16 的 `chunk` 取奇数时它带半个 word，取公约数无从谈起，只能按字节地址逐个算出线程落在哪条 bank 上再数。
2. **32 个线程请求的 $32w$ 个 word 互不相同**，即 $S \ge w$。落在同一个 word 上的线程走广播，不占额外的周期，下面按 $32w$ 个 word 计数就不成立；$S < w$ 时相邻线程的向量区间彼此重叠，同样不成立。

整个 warp 要取 $32w$ 个 word，而 shared memory 每周期最多处理 32 个，所以这条指令至少要 $w$ 个周期 —— 这是位宽带来的下界，与地址无关，冲突指的是超出这个下界的部分。$St \bmod 32$ 只取到 $g = \gcd(S, 32)$ 的倍数，也就是 $32/g$ 条 bank，每条被碰到 $g$ 次；再叠加 $j = 0, \dots, w-1$ 的平移。$g$ 与 $w$ 都是 2 的幂，必有一个整除另一个，于是分两种情形：

| | bank 的落点 | 周期数 | 冲突 |
| --- | --- | --- | --- |
| $g \le w$ | 32 条 bank 各被请求 $w$ 次 | $w$，正好是下界 | 无 |
| $g > w$ | 只有 $(32/g) \cdot w$ 条被碰到，各 $g$ 次 | $g$ | $g / w$ 路 |

$$\text{冲突路数} \ \ge\ \max\left(1,\ \frac{\gcd(S,\ 32)}{w}\right)$$

标量读（$w = 1$）时它就是 $\gcd(S, 32)$：$S$ 与 32 互素时 32 个线程铺满 32 条 bank，无冲突；$S$ 是 32 的倍数时全部挤在同一条 bank 上，32 路串行 —— fp16、`chunk = 64` 就是后者，$S$ 是 128 字节即 32 个 word。

写成不等式，是因为最后一步假定硬件能把任意一组无冲突的 word 凑进一个周期，而 NVIDIA 没有公开 64 bit 与 128 bit 访问时 lane 的实际分组方式。

### 用 pad 改变 stride

stride 由 `chunk` 决定，而 `chunk` 通常由算法决定，往往无法任意改动。当发生$N$路冲突时，一种可行方式是在每段末尾增加 `pad` 个元素，让 stride 变成 `chunk + pad` —— `gcd` 随之改变，冲突路数也就随之改变。例如：fp16、`chunk = 64`时，上只要加 2 个元素，stride 就从 32 个 word 变成 33 个，与 32 互素，冲突完全消失。

<figure class="bank-conflict" markdown="1">

<svg class="tf-bank" viewBox="0 0 520 340" role="img" aria-label="fp16、每段 64 个元素时，一个 warp 的 32 个线程落在 32 条 shared memory bank 上的分布。不加 pad 时全部落在 bank 0，是 32 路冲突；pad = 2 时铺满 32 条 bank，无冲突；pad = 4 时两个线程共用一条 bank，是 2 路；pad = 8 时四个共用一条，是 4 路。">
<text class="bk-title" x="0" y="43.0">pad = 0</text>
<text class="bk-sub" x="0" y="58.0">stride = 32 word</text>
<text class="bk-sub" x="0" y="72.0">gcd(32, 32) = 32</text>
<rect class="bk-cell bk-cell--many" x="150.0" y="30.0" width="11.0" height="20.0"/>
<text class="bk-count" x="155.5" y="43.5" text-anchor="middle">32</text>
<rect class="bk-cell" x="161.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="172.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="183.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="194.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="205.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="216.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="227.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="238.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="249.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="260.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="271.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="282.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="293.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="304.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="315.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="326.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="337.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="348.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="359.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="370.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="381.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="392.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="403.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="414.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="425.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="436.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="447.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="458.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="469.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="480.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="491.0" y="30.0" width="11.0" height="20.0"/>
<text class="bk-note" x="150.0" y="66.0">用到 1 / 32 条 bank，最多的一条要服务 32 个线程 —— 32 路冲突</text>
<text class="bk-title" x="0" y="117.0">pad = 2</text>
<text class="bk-sub" x="0" y="132.0">stride = 33 word</text>
<text class="bk-sub" x="0" y="146.0">gcd(33, 32) = 1</text>
<rect class="bk-cell bk-cell--one" x="150.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="161.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="172.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="183.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="194.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="205.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="216.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="227.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="238.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="249.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="260.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="271.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="282.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="293.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="304.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="315.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="326.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="337.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="348.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="359.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="370.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="381.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="392.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="403.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="414.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="425.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="436.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="447.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="458.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="469.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="480.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="491.0" y="104.0" width="11.0" height="20.0"/>
<text class="bk-note" x="150.0" y="140.0">用到 32 / 32 条 bank，最多的一条要服务 1 个线程 —— 无冲突</text>
<text class="bk-title" x="0" y="191.0">pad = 4</text>
<text class="bk-sub" x="0" y="206.0">stride = 34 word</text>
<text class="bk-sub" x="0" y="220.0">gcd(34, 32) = 2</text>
<rect class="bk-cell bk-cell--many" x="150.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="155.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="161.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="172.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="177.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="183.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="194.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="199.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="205.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="216.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="221.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="227.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="238.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="243.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="249.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="260.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="265.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="271.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="282.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="287.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="293.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="304.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="309.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="315.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="326.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="331.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="337.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="348.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="353.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="359.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="370.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="375.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="381.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="392.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="397.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="403.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="414.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="419.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="425.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="436.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="441.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="447.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="458.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="463.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="469.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="480.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="485.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="491.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-note" x="150.0" y="214.0">用到 16 / 32 条 bank，最多的一条要服务 2 个线程 —— 2 路冲突</text>
<text class="bk-title" x="0" y="265.0">pad = 8</text>
<text class="bk-sub" x="0" y="280.0">stride = 36 word</text>
<text class="bk-sub" x="0" y="294.0">gcd(36, 32) = 4</text>
<rect class="bk-cell bk-cell--many" x="150.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="155.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="161.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="172.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="183.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="194.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="199.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="205.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="216.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="227.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="238.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="243.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="249.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="260.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="271.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="282.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="287.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="293.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="304.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="315.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="326.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="331.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="337.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="348.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="359.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="370.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="375.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="381.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="392.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="403.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="414.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="419.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="425.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="436.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="447.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="458.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="463.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="469.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="480.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="491.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-note" x="150.0" y="288.0">用到 8 / 32 条 bank，最多的一条要服务 4 个线程 —— 4 路冲突</text>
<text class="bk-axis" x="155.5" y="22.0" text-anchor="middle">bank 0</text>
<text class="bk-axis" x="496.5" y="22.0" text-anchor="end">31</text>
<text class="bk-scale" x="0" y="337.0">fp16、每段 64 个元素、一个 warp 的 32 个线程。格内数字是落在该 bank 上的线程数，空格表示没有线程落上去。</text>
</svg>

<figcaption>四张图是同一个 kernel 在四个 pad 取值下，一个 warp 的 32 个线程在 32 条 bank 上的落点。格内数字是落在这条 bank 上的线程数，最大的那个数就是冲突路数。<code>pad = 0</code> 时 32 个线程全挤在 bank 0；加 2 个元素之后 stride 变成 33 个 word，与 32 互素，32 个线程正好铺满 32 条 bank。</figcaption>

</figure>

### pad 的选法

判据由上面的式子给出，**两条都要满足**：

| 访问位宽 | $w$ | 冲突路数要求 | 段起点的对齐要求 | fp32、`chunk = 64` 上最快的 pad |
| --- | --- | --- | --- | --- |
| 32 bit，标量读 | 1 | $\gcd(S, 32) = 1$ | 4 字节 | `pad = 1`（$S = 65$ word） |
| 64 bit，`float2` | 2 | $\gcd(S, 32) \le 2$ | 8 字节 | `pad = 2`（$S = 66$ word） |
| 128 bit，`float4` | 4 | $\gcd(S, 32) \le 4$ | 16 字节 | `pad = 4`（$S = 68$ word） |

两条方向相反，所以不能只看一条：**位宽越宽，对冲突路数的要求越松，对对齐的要求越紧**。奇数 pad 对标量读是最优解，对向量化读却是最差的 —— 它让段起点错开 4 字节。128 bit 的 shared 读要求 16 字节自然对齐，对齐不足时这条指令的行为是未定义的；编译期能看出对齐不足时，编译器会退回窄指令，下面实测里 $\gcd = 1$ 那一行就是这种情形。

实测（H200，fp32，`chunk = 64`，消费循环重复 32 遍使 shared 一侧成为瓶颈，耗时随遍数成比例）给出每种组合比同位宽最快的那一行慢多少倍：

| $\gcd(S, 32)$ | 32 bit | 64 bit | 128 bit | 下界预测（32 / 64 / 128） |
| --- | --- | --- | --- | --- |
| 1 | **1.00** | 1.10 | 2.27 | 1 / 1 / 1 |
| 2 | 1.72 | **1.00** | 1.01 | 2 / 1 / 1 |
| 4 | 3.29 | 1.81 | **1.00** | 4 / 2 / 1 |
| 8 | 6.52 | 3.55 | 4.04 | 8 / 4 / 2 |
| 16 | 12.97 | 7.62 | 7.99 | 16 / 8 / 4 |
| 32 | 25.82 | 13.97 | 15.71 | 32 / 16 / 8 |

对角线上的三个 1.00 就是上表那三行推荐。$g \le w$ 那一侧下界是紧的；$g > w$ 那一侧下界不紧，128 bit 实测恰好是它的 2 倍（4.04 对 2、7.99 对 4、15.71 对 8）；成因要更低层的证据才能定，NVIDIA 未公开这两种位宽下 lane 的分组方式。$\gcd = 1$ 那一行的 2.27 是对齐造成的：$S = 65$ word 即 260 字节，不是 16 的倍数。

### pad 随 chunk 变，不能写成常数 {#pad-per-chunk}

固定字节数的 pad 会在某些 `chunk` 上落回最坏情形。段起点的对齐要求把候选限制在 16 字节的整数倍，记 pad 为 $k$ 个 16 字节（$k = 1, 2, \dots$），于是

$$S = \frac{\text{chunk} \times E}{4} + 4k \ \text{word}$$

$k$ 固定时 $\gcd(S, 32)$ 随 `chunk` 变。fp16 与 bf16（$E = 2$）的几个 `chunk`：

| chunk | $S$（$k = 1$） | $\gcd(S, 32)$ | $S$（$k = 2$） | $\gcd(S, 32)$ |
| --- | --- | --- | --- | --- |
| 32 | 20 | **4** | 24 | 8 |
| 56 | 32 | 32 | 36 | **4** |
| 64 | 36 | **4** | 40 | 8 |
| 72 | 40 | 8 | 44 | **4** |
| 128 | 68 | **4** | 72 | 8 |

`chunk = 56` 那一行的 $S$ 恰好是 32 个 word，与 `pad = 0` 落在同一条 bank 上 —— pad 加了，冲突没消。做法是在 $k = 1$ 与 $k = 2$ 两个候选里取 $\gcd(S, 32)$ 小的那个。$\text{chunk} \times E$ 是 16 的倍数时，两者必有一个把 $\gcd$ 降到 4：记 $q = \text{chunk} \times E / 16$，则 $S = 4(q + k)$，$\gcd(S, 32) = 4 \gcd(q + k, 8)$，而 $q + 1$ 与 $q + 2$ 一奇一偶。

实测（H200，SM 时钟锁在 1830 MHz，bf16，CUPTI 设备耗时，每次迭代前清 L2，取 200 次的中位数，镜像 `ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev`）。四行的线程数都取到让 `chunk` 等于 56，同一行两列只差 pad：

| 输入 | 线程数 × chunk | $k = 1$（pad 8 个元素）<br>TB/s | $k = 2$（pad 16 个元素）<br>TB/s |
| --- | --- | --- | --- |
| $2048 \times 3584$ | 64 × 56 | 1.38 | **3.07**{ .win } |
| $2048 \times 7168$ | 128 × 56 | 1.58 | **3.29**{ .win } |
| $1024 \times 14336$ | 256 × 56 | 1.51 | **3.05**{ .win } |
| $512 \times 28672$ | 512 × 56 | 1.32 | **2.33**{ .win } |

这四个宽度是 Qwen2-7B 的 hidden size、Llama-3-70B 的 hidden size、Llama-3-8B 与 Llama-3-70B 的 FFN 中间维，不是构造出来的反例。

这组测量的 shared 一侧不是标量读：`cuobjdump` 显示 $S = 36$ word 时每线程 16 条 `LDS.64`，即 $w = 2$，所以路数是 $\gcd(S, 32) / 2$ —— 两列分别是 16 路与 2 路。路数差 8 倍而带宽只差 2.2 倍，是因为 2 路那一列的瓶颈已经回到 DRAM。段起点错开 8 字节以下时它退回 `LDS`（$w = 1$）—— 位宽由编译器定，不由声明 pad 的人定，所以上表两列都要实测，不能只算 $\gcd$。

### 实测

下面的实测每线程逐个元素读（$w = 1$），`chunk` 取 2 的幂，每个线程读自己那一段 —— 上面两个前提都成立。

H200，SM 时钟锁在 1830 MHz。fp16，输入 $65536 \times 4096$（512 MB，大于 60 MiB 的 L2）。kernel 把整行搬进 shared memory，每个线程一段 `chunk + pad` 个元素，逐段做串行前缀积再写回，读加写共 1 GB。括号内是上面公式预测的冲突路数：

| chunk | 线程数 | pad = 0<br>TB/s | pad = 2<br>TB/s | pad = 4<br>TB/s | pad = 8<br>TB/s | pad = 16<br>TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 256 | 1.63（8 路） | 2.99（1 路） | **3.29**{ .win }（2 路） | 2.58（4 路） | 0.86（16 路） |
| 32 | 128 | 0.89（16 路） | 3.32（1 路） | **3.35**{ .win }（2 路） | 2.57（4 路） | 1.54（8 路） |
| 64 | 64 | 0.46（32 路） | 3.63（1 路） | **3.70**{ .win }（2 路） | 2.74（4 路） | 1.62（8 路） |
| 128 | 32 | 0.46（32 路） | 3.25（1 路） | **3.31**{ .win }（2 路） | 2.76（4 路） | 1.61（8 路） |

20 个配置里，1 路与 2 路在 3.0 以上，4 路降到 2.6 附近，8 路及以上跌到 1.6 以下。带宽随预测的路数单调下降，所以在这组条件下公式可以直接用来缩小 pad 的候选。

### 取舍

1. **公式给出候选，最终值靠实测。** 上表里 1 路与 2 路都在 3.0 以上，4 路降到 2.6 附近，8 路及以上跌到 1.6 以下，所以公式的用处是把候选缩到「算出来不超过 $w$ 路」的那几个。这几个之间要实测：四组 `chunk` 上 2 路都略高于 1 路，但差距从 0.9% 到 10% 不等，不构成一条可以照搬的规则。

2. **改完 pad 要重扫 `chunk`。** `pad = 0` 那一列最优的是 `chunk = 16`（1.63），`pad = 4` 那一列最优的是 `chunk = 64`（3.70）。前一列的排序主要由冲突路数决定 —— `chunk = 16` 撞的是 8 路，`chunk = 64` 撞的是 32 路。冲突消掉之后四组都是 2 路，最优点换了位置。表里线程数与 `chunk` 联动（两者之积恒为行宽 4096），所以换位置的成因不止 `chunk` 一个，占用率与循环长度也跟着在变；能确定的只是改完 pad 之后 `chunk` 的排序会变。

3. **`chunk` 变了要重算 pad。** pad 写成固定字节数，等于假定 $\gcd(S, 32)$ 与 `chunk` 无关，而 [pad 随 `chunk` 变](#pad-per-chunk) 那张表说明它不是：`chunk = 56`、$k = 1$ 时 $S$ 回到 32 个 word。把 pad 写成 `chunk` 的函数 —— 在 $k = 1, 2$ 里取 $\gcd(S, 32)$ 小的那个 —— 才和这一条自洽。

4. **这一条只在数据经过 shared memory 时适用。** 按[第 1 条](#coalescing)的取舍，$V$ 小时用向量化的 blocked，数据直接进寄存器，不经 shared memory，没有 bank 冲突可言；$V$ 大到寄存器压力压低占用率时才改用 staged，以及整行需要被 block 内所有线程共享时，这一条才适用。

下面两段是 shared 缓冲的声明，差别只在 stride。反例，stride 恰好是 32 个 word 的倍数：

```python
sh = T.alloc_shared((threads * chunk,), dtype)          # stride = chunk 个元素
```

正例，pad 按 `chunk` 算，取 16 字节的整数倍里 $\gcd(S, 32)$ 最小的那个：

```python
import math

def pick_pad(chunk: int, elem_bytes: int) -> int:
    """16 字节的整数倍里，gcd(S, 32) 最小的 pad，单位是元素。"""
    return min(
        (16 // elem_bytes, 32 // elem_bytes),                    # k = 1、k = 2
        key=lambda pad: math.gcd((chunk + pad) * elem_bytes // 4, 32),
    )

pad = pick_pad(chunk, elem_bytes)                                # 候选之间仍要实测
sh = T.alloc_shared((threads * (chunk + pad),), dtype)           # stride = chunk + pad
```

## 3. 用 T.copy 把整行读进 fragment {#registers}

reduction 类算子通常把输入看作二维，一个 block 负责其中一行 —— 例如按行求最大值。这一行的每个元素都要被这个 block 读一遍，access pattern 有三种：

1. 逐个元素从 global memory 读；
2. 一次 `T.copy` 把整行搬进 fragment，再遍历；
3. 经 shared memory 中转，即[第 1 条](#coalescing)里的 staged。

第三种在第 1 条测过，胜负取决于每线程持有的元素数：不超过 32 个时直接进寄存器更快，到 64 个以上时加了 pad 的 staged 反超（串行前缀上 3.69 对 1.80 TB/s），因为寄存器压力上来之后占用率降了。这一条比的是前两种。

`T.copy` 进 fragment 是**从 global memory 直接进线程私有寄存器**，不经 shared memory。读生成的 CUDA 可以确认：`frag` 声明为 `half_t frag[32]`，整个 kernel 没有 `__shared__`，搬运语句是 `*(uint4*)(frag + i * 8) = *(uint4*)(X + ...)`，即一条 16 字节的向量访问。

线程映射由 layout inference 给出，是合并的：固定迭代序号时相邻线程的地址相隔 16 字节，一个 warp 的一条指令覆盖 512 连续字节。**但每个线程持有的不是连续的一段** —— 行宽 4096、128 线程时，每线程拿到 4 块各 8 个元素，块与块之间相隔 1024 个元素。因此要求每线程持有连续一段的计算（例如串行前缀）用不了这条路线，那种情形见[第 1 条](#coalescing)。

还有一个代码上的差别。逐个元素读那种 access pattern 要自己算下标 —— `index = it * threads + tx` —— 并且要加 `if index < N` 的边界检查，否则最后一轮会越界。遍历 fragment 时循环变量本身就是列下标，两样都不需要。下面的代码可以对照。

### 实测

H200，SM 时钟锁在 1830 MHz，fp16，输入 $65536 \times 4096$（512 MB，大于 60 MiB 的 L2）。kernel 求每行的最大值：反例逐个元素从 global memory 读，正例先 `T.copy` 进 fragment。行宽固定、只变 block 的线程数，因此每线程持有的元素数是唯一变量：

| 行宽 | 线程数 | 每线程元素 | 每线程寄存器 | 逐元素<br>TB/s | `T.copy` 进 fragment<br>TB/s |
| --- | --- | --- | --- | --- | --- |
| 4096 | 512 | 8 | 4 | 3.11 | **3.17**{ .win } |
| 4096 | 256 | 16 | 8 | 3.38 | **4.24**{ .win } |
| 4096 | 128 | 32 | 16 | 3.70 | **4.38**{ .win } |
| 4096 | 64 | 64 | 32 | 3.97 | **4.35**{ .win } |
| 4096 | 32 | 128 | 64 | 3.83 | **4.33**{ .win } |
| 16384 | 128 | 128 | 64 | 3.69 | **4.44**{ .win } |
| 16384 | 64 | 256 | 128 | 3.74 | **4.39**{ .win } |

fragment 一侧在 8 到 128 个寄存器之间基本平坦，`4.24` 到 `4.44` TB/s；只在 512 线程那一行退回与逐元素同速。

### 判断寄存器是否够用

每线程 255 个 32 位寄存器是架构上限，超过之后编译器把放不下的部分挪到 local memory。但**这件事本身不一定有代价**：

| 每线程元素 | 每线程寄存器 | 遍历 1 遍<br>TB/s | 遍历 4 遍<br>TB/s |
| --- | --- | --- | --- |
| 16 | 8 | **4.04**{ .win } | 3.78 |
| 64 | 32 | **4.20**{ .win } | 4.01 |
| 256 | 128 | **4.25**{ .win } | 3.73 |
| 512 | **256** | **4.05**{ .win } | 2.62 |

每线程 512 个 fp16 元素折合 256 个寄存器，已经越过上限，生成的 CUDA 里确实是一个 `[512]` 的数组；只遍历一遍时带宽仍有 4.05 TB/s，遍历四遍才掉到 2.62。原因是 local memory 的访问本身是合并的、且经 L1 与 L2 缓存，遍历一遍时它的延迟被 DRAM 侧掩掉；反复遍历才把它暴露成瓶颈。

### 取舍

**优先 `T.copy` 进 fragment。** 除了 512 线程那一档，它在所有配置上都比逐元素快 0.5 到 1.0 TB/s。

**不必为了「装得进寄存器」去压线程数。** 上面 8 到 128 个寄存器的区间里带宽是平的，甚至越过 255 上限也只在 fragment 被反复遍历时才掉。线程数取 64 到 256 之间都可以，512 明显偏大。

**fragment 要被反复遍历时，才需要核对寄存器数量。** 判断依据是「每线程持有的 32 位量 = 元素数 × 元素字节 / 4」，越过 255 且要多次遍历时，改成分块处理或退回 shared memory（[第 2 条](#bank-conflict)）。

下面两段读的是同一行数据，差别在于数据落在哪里。反例，逐个元素从 global memory 读：

```python
for it in T.serial(iterations):
    index = it * threads + tx
    if index < N:
        update(best, key_of(x[row, index]), index)
```

正例，一次 `T.copy` 进 fragment，列下标就是循环变量：

```python
row_frag = T.alloc_fragment((1, N), dtype)
T.copy(x[row : row + 1, :], row_frag)

for _, column in T.Parallel(1, N):
    update(best, key_of(row_frag[0, column]), column)
```

## 4. 让输入与输出都用满一次向量访问 {#narrow-output}

谓词类算子的输出 dtype 比输入窄，例如 fp32 输入、1 字节输出。同一个每线程元素数在输入侧能用满一次向量访问，在输出侧就填不满。要让两侧都用满，需要同时做两件事：提高每线程元素数，并让这个宽度成为一条向量访问。只做其中一件会比不做更慢。

每线程 16 字节是 PTX 一次向量访问的最大宽度（SASS 里是 `LDG.E.128`），32 个线程 × 16 字节 = 512 字节，正好 4 条 128 字节 cache line。按输入算，fp32 取 4 个元素刚好用满。但 1 字节的输出此时每线程只有 4 字节，存储侧远没有用满一次访问的宽度 —— 所以想让存储侧也划算，就得让每线程多取几个元素。

**而 bool 输出会把这条路堵死。** CUDA 没有 8 个 bool 的向量类型，codegen 发不出对应的存储，编译当场失败：

```
Fatal: Cannot convert type boolx8 to CUDA type
```

改用 int8 存储即可绕过：`torch.bool` 每个元素占 1 字节，而 `T.if_then_else` 已把结果规范成 0 或 1，两种 dtype 的字节模式逐字节相同，所以 `.view(torch.bool)` 不拷贝数据。（前提正是这个规范化；把任意非零字节当 bool 看是不安全的。）

### 实测

H200，SM 时钟锁在 1830 MHz。`isnan`：fp32 输入 134217728 个元素（512 MB，大于 60 MiB 的 L2），1 字节输出，读加写共 640 MB。同一个算子写成两种形态 —— 显式 `T.vectorized` 读写本地数组，或者纯 `T.serial` 逐元素：

| 每线程元素 | 输出 dtype | `T.vectorized`<br>TB/s | `T.serial`<br>TB/s |
| --- | --- | --- | --- |
| 4 | int8 | **4.12**{ .win } | 4.07 |
| 8 | int8 | **4.28**{ .win } | 2.89 |
| 16 | int8 | **4.28**{ .win } | 1.16 |
| 4 | bool | **4.11**{ .win } | 4.08 |
| 8 | bool | 编译失败 | 2.87 |
| 16 | bool | 编译失败 | 1.16 |

两栏的走向相反，这是这一条的全部内容：

- **向量化那一栏**：每线程从 4 个加到 8 个，4.12 → 4.28，涨 4%；再加到 16 个不再涨。
- **未向量化那一栏**：每线程从 4 个加到 16 个，4.07 → 1.16，**慢 3.5 倍**。
- **bool 输出在向量化那一栏拿不到 8 和 16** —— 编译不过，所以它只能停在 4.11。

也就是说，「每线程多取」本身不是收益，**与向量化配套才是**；而 int8 存储的作用是让 bool 结果也能进入向量化那一栏。

### 取舍

**每线程元素数与向量化一起改，不要单独加。** 单独把元素数加到 16 而不向量化，比不动更差三倍多。

**谓词类算子用 int8 存储，返回时 `.view(torch.bool)`。** 这不是为了省字节（两者都是 1 字节），而是为了让向量化那条路走得通。

**收益有限，别期待太多。** 4.11 → 4.28 是 4%，而向量化本身相对标量是 4.07 → 1.16 反向的三倍多。先确保向量化，再谈这一条。

下面两段的差别在每线程元素数怎么定、以及输出用什么 dtype。反例，元素数按输入宽度定，且输出是 bool：

```python
npt = _BYTES_PER_THREAD // elem_bytes        # 只看输入宽度

def op_func(x):
    return T.isnan(T.cast(x, "float32"))     # bool 输出，向量化到 8 就编译不过
```

正例，元素数加倍并用 int8 承接：

```python
_BOOL_STORAGE_DTYPE = "int8"

npt = _BYTES_PER_THREAD // elem_bytes
if stored_bytes < elem_bytes:
    npt *= 2                                 # 输出更窄，每线程多取一次输入

def wrapped(x):
    return T.if_then_else(
        op_func(x),
        T.cast(1, _BOOL_STORAGE_DTYPE),
        T.cast(0, _BOOL_STORAGE_DTYPE),
    )

# forward 里把 int8 结果交回为 bool，不拷贝
return result.view(torch.bool)
```
