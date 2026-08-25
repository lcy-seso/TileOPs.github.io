# 优化 access pattern

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

    缓存查找时以 cache line 为单位，搬运数据时以 sector 为单位：某个 sector 未命中，L1 就只向 L2 请求这一个 sector，不必把整条 cache line 都拉过来。由此得到的结论是**取 1 个字节和取满 32 个字节的代价相同**。于是一条访存指令的好坏由 **sector 利用率**衡量：`真正用到的字节 / (覆盖的 sector 数 × 32)`。


- **shared memory 由 32 条 bank 构成，每条 bank 宽 4 字节，32 条可以同时被访问。** 一个地址落在哪条 bank 上，由 `(字节地址 / 4) mod 32` 决定。多个线程并发访问 shared memory 时，只要它们落在不同的 bank 上，这些请求就能被并行地服务完。

    落在同一条 bank 上则不然。同一个 warp 内若有多个线程访问一条 bank 上的**不同** word，硬件会把这次请求拆成若干次无冲突的请求依次完成，拆分的次数就是冲突的路数。例外是**读取同一个** word：任意两个线程只要落在同一个 word 内（哪怕取的是其中不同的字节），彼此就不冲突，这个 word 会被广播给所有请求它的线程；分布在不同 bank 上的多个广播还会合并为一次 multicast。这两种情形都不产生冲突。（若多个线程写入同一地址，只有一个写入生效，是哪一个未定义。）

## 1. 合并 global memory 的访存 {#coalescing}

硬件把一个 warp 的 32 个访问合并成尽可能少的 32 字节事务。**地址连续、按 32 字节对齐、每个线程一次取满 16 字节** —— 这三个因素同时成立时事务数最少，对硬件最友好。

一个线程要读 $V$ 个元素时（$V$ = 行宽 / 线程数），有四种写法。

**blocked** —— 每个线程负责一段连续的元素。固定 `c` 时相邻线程的地址相隔 $V$ 个元素，违反第一个因素，sector 利用率是 $1/V$：

```python
for c in T.serial(V):
    acc[0] = acc[0] * T.cast(X[row, tx * V + c], "float32")
```

**striped** —— 相邻线程取相邻元素。地址连续了，但每个线程一次只取一个元素，违反第三个因素，$V$ 个元素要发 $V$ 条指令：

```python
for c in T.serial(V):
    acc[0] = acc[0] * T.cast(X[row, c * threads + tx], "float32")
```

**blocked + 向量化** —— 仍是每线程一段连续的，但改用 `T.vectorized` 一次读满 16 字节，三个因素全部满足：

```python
buf = T.alloc_local((V,), dtype)

for c in T.vectorized(V):
    buf[c] = X[row, tx * V + c]
for c in T.serial(V):
    acc[0] = acc[0] * T.cast(buf[c], "float32")
```

**staged** —— 搬运交给 `T.Parallel`，消费改从 shared memory 读，同样满足三个因素：

```python
sh = T.alloc_shared((threads, V + pad), dtype)

for t, c in T.Parallel(threads, V):
    sh[t, c] = X[row, t * V + c]
T.sync_threads()
for c in T.serial(V):
    acc[0] = acc[0] * T.cast(sh[tx, c], "float32")
```

四种写法的差别归结为一件事：**「哪个线程读哪些元素、一次读多宽」这个映射由谁决定。**

`T.serial` 的语义是循环体由单个线程顺序执行，索引表达式被逐字翻译成访存指令，不做合并也不做向量化 —— 编程者写出的模式就是硬件看到的模式。

`T.vectorized`、`T.Parallel`、`T.copy` 则由 TileLang 的 **layout inference** 决定，三者的差别在于编程者还需要写明多少：`T.vectorized` 要写明每线程一次访问的宽度，线程映射由 layout inference 推导；`T.Parallel` 连宽度也不必写，循环维度怎么分给线程、一次读多宽都由它决定；`T.copy` 只写源和目标两个区域，整段搬运由它生成（需要接管推导结果时，另有 `coalesced_width` 与 `loop_layout` 两个参数）。剩下的向量化、地址对齐、以及在 shared memory 一侧避开 bank 冲突，都由它负责 —— 这些正是对硬件友好但手写容易出错的部分。

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

H200，锁频 1830 MHz，bf16，每行 4096 个元素，输入 512 MB（**必须大于 L2 的 60 MiB**，否则测到的是 L2 带宽）。单位 TB/s，三次跑一致到 ±0.5%。staged 给两个数：不加 pad（stride 撞上 bank 冲突，见[第 2 条](#bank-conflict)）与扫 pad 后的最优值。

按行 `prod`，与顺序无关，只读：

| 线程数 | $V$ | blocked<br>TB/s | 向量化<br>TB/s | striped<br>TB/s | staged 不加 pad<br>TB/s | staged 加 pad<br>TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 8 | 3.02 | **3.22**{ .win } | 3.04 | 3.05 | 3.02 |
| 256 | 16 | 1.83 | **3.81**{ .win } | 3.31 | 3.79 | 3.79 |
| 128 | 32 | 0.95 | **3.99**{ .win } | 3.36 | 3.81 | 3.98 |
| 64 | 64 | 0.48 | 3.32 | 3.49 | 2.80 | **4.02**{ .win } |
| 32 | 128 | 0.47 | 3.11 | 3.43 | 2.83 | **3.88**{ .win } |

按行的串行前缀积，顺序受约束，读加写。striped 在这里不可用 —— 它不让线程持有连续的一段：

| 线程数 | $V$ | blocked<br>TB/s | 向量化<br>TB/s | staged 不加 pad<br>TB/s | staged 加 pad<br>TB/s |
| --- | --- | --- | --- | --- | --- |
| 512 | 8 | 0.92 | **4.20**{ .win } | 2.47 | 3.15 |
| 256 | 16 | 0.42 | **3.85**{ .win } | 1.63 | 3.29 |
| 128 | 32 | 0.34 | 2.76 | 0.89 | **3.35**{ .win } |
| 64 | 64 | 0.26 | 1.80 | 0.46 | **3.69**{ .win } |
| 32 | 128 | 0.27 | 1.60 | 0.46 | **3.24**{ .win } |

### 取舍

1. **不要用逐元素的 blocked。** $V$ 越大越差：$V = 8$ 时 3.02，$V = 64$ 时只剩 0.48。
2. **$V \le 32$ 用向量化的 blocked。** 3.22 / 3.81 / 3.99，且不需要 shared memory 与同步。
3. **$V \ge 64$ 改用 staged 并加 pad。** 向量化要占每线程 $V/2$ 个寄存器，占用率随之下降；staged 不吃寄存器，加 pad 后 4.02 / 3.88。
4. **staged 一定要加 pad。** 不加时 scan 上 $V = 64$ 是 0.46 对 3.69。
5. **顺序不受约束时 striped 是改动最小的修法** —— 只改一个索引表达式，从 1.83 提到 3.31。

### 代码

$V \le 32$，向量化的 blocked：

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

$V \ge 64$，加了 pad 的 staged：

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

## 2. 消除 shared memory 的 bank 冲突 {#bank-conflict}

[第 1 条](#coalescing)的 staged 路线把整行搬进 shared memory，每个线程再读自己那一段。消费时固定段内偏移，一个 warp 的 32 个线程相隔一整段 —— 这个 stride 决定它们落在几条 bank 上。

设一段有 `chunk` 个元素、元素 `E` 字节，则 stride 折成 4 字节 word 是 `chunk * E / 4` 个。落到多少条不同的 bank 上由它与 32 的最大公约数决定：

```
不同的 bank 数 = 32 / gcd(stride_words, 32)
冲突路数       = gcd(stride_words, 32)
```

`gcd` 等于 1 时 32 个线程铺在 32 条 bank 上，无冲突；等于 32 时全部挤在同一条 bank 上，32 路串行。fp16、`chunk = 64` 就是后者：stride 是 128 字节即 32 个 word，`gcd(32, 32) = 32`。

在 stride 上加 `pad` 个元素即可改变这个公约数。`gcd` 为 1（即 `(chunk + pad) * E / 4` 为奇数）时冲突完全消失，为 2 时退化到 2 路 —— 实测这两档都好，4 路及以上都差。

### 实测

H200，SM 时钟锁在 1830 MHz。fp16，每行 4096 个元素，行数取 65536 使输入为 512 MB（大于 60 MiB 的 L2）。kernel 把整行搬进 `(threads, chunk + pad)` 的 shared 缓冲，逐段做串行前缀积再写回，读加写共 1 GB。括号内是上面公式预测的冲突路数：

| chunk | 线程数 | pad = 0<br>TB/s | pad = 2<br>TB/s | pad = 4<br>TB/s | pad = 8<br>TB/s | pad = 16<br>TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 256 | 1.63（8 路） | 2.99（1 路） | **3.29**{ .win }（2 路） | 2.58（4 路） | 0.86（16 路） |
| 32 | 128 | 0.89（16 路） | 3.32（1 路） | **3.35**{ .win }（2 路） | 2.57（4 路） | 1.54（8 路） |
| 64 | 64 | 0.46（32 路） | 3.63（1 路） | **3.70**{ .win }（2 路） | 2.74（4 路） | 1.62（8 路） |
| 128 | 32 | 0.46（32 路） | 3.25（1 路） | **3.31**{ .win }（2 路） | 2.76（4 路） | 1.61（8 路） |

20 个配置里，带宽随预测的路数单调下降 —— 1 路与 2 路在 3.0 以上，4 路降到 2.6 附近，8 路及以上跌到 1.6 以下。公式可以直接用来选 pad，不必逐个试。

### 取舍

**用公式缩小候选，再实测挑一个 —— 公式不直接给出最优值。** 上表里 1 路与 2 路都很好、4 路及以上都很差，所以公式的用处是把 pad 的候选缩到「算出来是 1 路或 2 路」的那几个。但**四组 chunk 里 2 路一律略好于 1 路**（3.29 对 2.99、3.35 对 3.32、3.70 对 3.63、3.31 对 3.25），所以在这两个候选之间要实测，不能只看公式。

**不要按字节对齐选。** `pad = 8`（16 字节）落在 4 路，`pad = 16`（32 字节）落在 8 路甚至 16 路。16 字节对齐能保住 shared memory 一侧的向量访问，但它引入的冲突远大于这点收益：`pad = 8` 的 2.57 到 2.76 明显低于 `pad = 4` 的 3.29 到 3.70。

**pad 之后最优的 chunk 会变。** `pad = 0` 时最优是 `chunk = 16`（1.63），`pad = 4` 时最优是 `chunk = 64`（3.70）。冲突消掉之后，更大的段意味着更少线程争用同一批 bank，最优点因此往大移。改完 pad 要重扫 chunk。

**这一条只在 staged 路线上出现。** 第 1 条的实测表明，向量化的 blocked 通常更快，那条路线把数据放进寄存器，不经 shared memory，也就没有 bank 冲突可言。只有整行需要被 block 内所有线程共享时才走 staged，届时这一条适用。

### 代码

反例，stride 恰好是 32 个 word 的倍数：

```python
staged = T.alloc_shared((threads, chunk), dtype)   # stride = chunk
```

正例，pad 到 stride 的 word 数为奇数：

```python
# fp16：pad = 2 算出来是 1 路，pad = 4 是 2 路，两者都可用；
# 上表四组 chunk 里 pad = 4 一律略好，所以取 4，换形状时重扫一遍
pad = 4
staged = T.alloc_shared((threads, chunk + pad), dtype)
```

## 3. 用 T.copy 把整行读进 fragment {#registers}

reduction 类算子通常把输入看作二维，一个 block 负责其中一行 —— 例如按行求最大值。这一行的每个元素都要被这个 block 读一遍，写法有三种：

1. 逐个元素从 global memory 读；
2. 一次 `T.copy` 把整行搬进 fragment，再遍历；
3. 经 shared memory 中转，即[第 1 条](#coalescing)里的 staged。

第三种在第 1 条测过，胜负取决于每线程持有的元素数：不超过 32 个时直接进寄存器更快，到 64 个以上时加了 pad 的 staged 反超（串行前缀上 3.69 对 1.80 TB/s），因为寄存器压力上来之后占用率降了。这一条比的是前两种。

`T.copy` 进 fragment 是**从 global memory 直接进线程私有寄存器**，不经 shared memory。读生成的 CUDA 可以确认：`frag` 声明为 `half_t frag[32]`，整个 kernel 没有 `__shared__`，搬运语句是 `*(uint4*)(frag + i * 8) = *(uint4*)(X + ...)`，即一条 16 字节的向量访问。

线程映射由 layout inference 给出，是合并的：固定迭代序号时相邻线程的地址相隔 16 字节，一个 warp 的一条指令覆盖 512 连续字节。**但每个线程持有的不是连续的一段** —— 行宽 4096、128 线程时，每线程拿到 4 块各 8 个元素，块与块之间相隔 1024 个元素。因此要求每线程持有连续一段的计算（例如串行前缀）用不了这条路线，那种情形见[第 1 条](#coalescing)。

还有一个写法上的差别。逐个元素读那种写法要自己算下标 —— `index = it * threads + tx` —— 并且要加 `if index < N` 的边界检查，否则最后一轮会越界。遍历 fragment 时循环变量本身就是列下标，两样都不需要。下面的代码可以对照。

### 实测

H200，SM 时钟锁在 1830 MHz，fp16，输入 512 MB（大于 60 MiB 的 L2）。kernel 求每行的最大值：反例逐个元素从 global memory 读，正例先 `T.copy` 进 fragment。行宽固定、只变 block 的线程数，因此每线程持有的元素数是唯一变量：

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

### 代码

反例，逐个元素从 global memory 读：

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

H200，SM 时钟锁在 1830 MHz。`isnan`：fp32 输入 512 MB（大于 60 MiB 的 L2），1 字节输出，读加写共 640 MB。同一个算子写成两种形态 —— 显式 `T.vectorized` 读写本地数组，或者纯 `T.serial` 逐元素：

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

### 代码

反例，元素数按输入宽度定，且输出是 bool：

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
