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

## 1. 让 global memory 的访存合并 {#coalescing}

一个线程要消费 $V$ 个元素时，「哪个线程读哪些元素」这个映射有四种写法。

<figure class="access-patterns" markdown="1">

<svg class="tf-access" viewBox="0 0 520 330" role="img" aria-label="三种访存模式下，一个 warp 的一条读取指令触及的元素，以及硬件因此取回的 sector。blocked 覆盖 4 个 sector，每个只用到 2 个元素；向量化后的 blocked 与 striped 都完全合并；staged 的搬运阶段与 striped 一样合并，消费阶段在 shared memory 上，不存在 sector。">
<text class="ap-title" x="0" y="24.0">blocked</text>
<text class="ap-sub" x="0" y="38.0">tx * V + c</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="14.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="229.0" y="14.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="322.0" y="14.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="415.0" y="14.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-cell" x="136.0" y="14.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="25.0" r="3.2"/>
<rect class="ap-cell" x="147.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="158.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="169.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="180.0" y="14.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="25.0" r="3.2"/>
<rect class="ap-cell" x="191.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="202.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="213.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="229.0" y="14.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="25.0" r="3.2"/>
<rect class="ap-cell" x="240.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="251.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="262.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="273.0" y="14.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="25.0" r="3.2"/>
<rect class="ap-cell" x="284.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="295.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="306.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="322.0" y="14.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="25.0" r="3.2"/>
<rect class="ap-cell" x="333.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="344.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="355.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="366.0" y="14.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="25.0" r="3.2"/>
<rect class="ap-cell" x="377.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="388.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="399.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="415.0" y="14.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="25.0" r="3.2"/>
<rect class="ap-cell" x="426.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="437.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="448.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="459.0" y="14.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="25.0" r="3.2"/>
<rect class="ap-cell" x="470.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="481.0" y="14.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="492.0" y="14.0" width="11.0" height="22.0"/>
<text class="ap-note" x="136.0" y="51.0">取回 4 个 sector，每个只用到 2 / 8 个元素</text>
<text class="ap-title" x="0" y="94.0">blocked + 向量化</text>
<text class="ap-sub" x="0" y="108.0">T.vectorized(V)</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="84.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="229.0" y="84.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="322.0" y="84.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="415.0" y="84.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-cell" x="136.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="147.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="158.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="169.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="180.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="191.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="202.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="213.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="229.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="240.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="245.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="251.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="256.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="262.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="267.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="273.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="284.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="289.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="295.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="300.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="306.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="311.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="322.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="333.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="338.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="344.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="349.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="355.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="360.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="366.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="377.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="382.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="388.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="393.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="399.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="404.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="415.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="426.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="431.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="437.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="442.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="448.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="453.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="459.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="470.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="475.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="481.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="486.5" cy="95.0" r="3.2"/>
<rect class="ap-cell" x="492.0" y="84.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="497.5" cy="95.0" r="3.2"/>
<text class="ap-note" x="136.0" y="121.0">一条 16 字节向量读，取回 4 个 sector，全部用满</text>
<text class="ap-title" x="0" y="164.0">striped</text>
<text class="ap-sub" x="0" y="178.0">c * threads + tx</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="154.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector" x="229.0" y="154.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector" x="322.0" y="154.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector" x="415.0" y="154.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-cell" x="136.0" y="154.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="165.0" r="3.2"/>
<rect class="ap-cell" x="147.0" y="154.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="165.0" r="3.2"/>
<rect class="ap-cell" x="158.0" y="154.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="165.0" r="3.2"/>
<rect class="ap-cell" x="169.0" y="154.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="165.0" r="3.2"/>
<rect class="ap-cell" x="180.0" y="154.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="165.0" r="3.2"/>
<rect class="ap-cell" x="191.0" y="154.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="165.0" r="3.2"/>
<rect class="ap-cell" x="202.0" y="154.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="165.0" r="3.2"/>
<rect class="ap-cell" x="213.0" y="154.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="165.0" r="3.2"/>
<rect class="ap-cell" x="229.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="240.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="251.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="262.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="273.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="284.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="295.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="306.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="322.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="333.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="344.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="355.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="366.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="377.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="388.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="399.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="415.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="426.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="437.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="448.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="459.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="470.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="481.0" y="154.0" width="11.0" height="22.0"/>
<rect class="ap-cell" x="492.0" y="154.0" width="11.0" height="22.0"/>
<text class="ap-note" x="136.0" y="191.0">取回 1 个 sector，8 / 8 个元素全部用到</text>
<text class="ap-title" x="0" y="234.0">staged</text>
<text class="ap-sub" x="0" y="248.0">T.Parallel 整行搬运</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="224.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="229.0" y="224.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="322.0" y="224.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="415.0" y="224.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-cell" x="136.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="147.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="158.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="169.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="180.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="191.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="202.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="213.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="229.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="240.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="245.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="251.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="256.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="262.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="267.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="273.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="284.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="289.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="295.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="300.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="306.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="311.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="322.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="333.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="338.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="344.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="349.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="355.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="360.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="366.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="377.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="382.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="388.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="393.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="399.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="404.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="415.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="426.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="431.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="437.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="442.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="448.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="453.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="459.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="470.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="475.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="481.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="486.5" cy="235.0" r="3.2"/>
<rect class="ap-cell" x="492.0" y="224.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="497.5" cy="235.0" r="3.2"/>
<text class="ap-note" x="136.0" y="261.0">取回 4 个 sector，全部用满</text>
<text class="ap-sub" x="0" y="296.0">staged[tx * V + c]</text>
<rect class="ap-sector ap-sector--onchip" x="136.0" y="272.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--onchip" x="229.0" y="272.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--onchip" x="322.0" y="272.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--onchip" x="415.0" y="272.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-cell" x="136.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="147.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="158.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="169.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="180.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="191.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="202.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="213.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="229.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="240.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="245.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="251.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="256.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="262.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="267.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="273.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="284.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="289.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="295.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="300.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="306.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="311.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="322.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="333.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="338.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="344.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="349.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="355.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="360.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="366.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="377.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="382.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="388.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="393.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="399.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="404.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="415.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="426.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="431.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="437.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="442.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="448.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="453.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="459.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="470.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="475.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="481.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="486.5" cy="283.0" r="3.2"/>
<rect class="ap-cell" x="492.0" y="272.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="497.5" cy="283.0" r="3.2"/>
<text class="ap-note" x="136.0" y="309.0">在 shared memory 上消费，不存在 sector</text>
<text class="ap-scale" x="0" y="326.0">示意图：fp32、8 个线程、V = 4，一个 sector 装 8 个元素。正文实测用 bf16、256 线程、V = 16，形状相同。</text>
</svg>

<figcaption>一个 warp 的一条读取指令：紫点是它真正读到的元素，青色底是硬件因此必须取回的 sector。blocked 摊开在 4 个 sector 上而每个只取用其中两个元素；向量化之后同样的地址由一条 16 字节访问一次读完，4 个 sector 全部用满；striped 把 32 个请求收进 1 个 sector；staged 的搬运阶段与 striped 同样合并，消费阶段落在 shared memory 上，那里不按 sector 组织，虚线框表示这一层不再有合并问题。</figcaption>

</figure>

选哪一种，要同时看两个互相拉扯的要求：

- **计算的要求**：有些计算必须让一个线程持有连续的一段，例如串行前缀，它的第 `c` 个结果依赖自己的第 `c-1` 个。
- **硬件的要求**：同一条指令里 32 个线程的地址越挨在一起，覆盖的 sector 越少，sector 利用率越高。

下表中间两列分别对应这两个要求：

| 访存模式 | 每个线程持有 | 一条指令里 32 个线程的地址 | sector 利用率 |
| --- | --- | --- | --- |
| **blocked** | 连续的一段，`x[row, tx * V + c]` | 两两相隔 $V$ 个元素 | 6.25% |
| **blocked + 向量化** | 连续的一段，同上 | 一条指令读满自己那一段，32 段首尾相接 | 100% |
| **striped** | 交错的单个元素，`x[row, c * threads + tx]` | 两两相邻 | 100% |
| **staged** | 连续的一段，`staged[tx * V + c]` | 搬运阶段两两相邻，由 `T.Parallel` 给出 | 100% |

只有 blocked 两个要求都不满足。它的 sector 利用率是 $1/V$ —— 一个 warp 的一条指令用到 32 个线程各 2 字节共 64 字节，却铺开在 $64V$ 字节上 —— 取 bf16、$V = 16$ 时是 6.25%。其余三种都完全合并。

**但 sector 利用率衡量的是单条指令，不能直接当作带宽的预测量。** blocked 的 `c` 循环会把同一批 sector 反复走一遍：第一次迭代把它们取进 L1，后续迭代若仍命中，冗余就不会传到 DRAM。这个重用窗口是每 warp $64V$ 字节，$V$ 越大越留不住。下面的实测里，$V = 8$ 的 blocked 与完全合并的写法基本同速，而 $V$ 增大后逐级掉下去。

### 实测

H200，SM 时钟锁在 1830 MHz。每行 4096 个 bf16 元素，行数取 65536，使输入为 512 MB —— **必须大于 L2 才测得到 DRAM 带宽**，这块卡的 L2 是 60 MiB（`cudaDeviceProp.l2CacheSize` = 62914560 字节）。$V$ 由 block 的线程数决定，行宽与输入大小始终不变，因此只有每线程的寄存器需求在变。每个数字三次跑一致到 ±0.5%。

按行 `prod`，与顺序无关，只读 512 MB：

| 线程数 | $V$ | blocked | blocked + 向量化 | striped | staged |
| --- | --- | --- | --- | --- | --- |
| 512 | 8 | 3.02 | **3.19** | 3.07 | 3.05 |
| 256 | 16 | 1.83 | 3.84 | 3.35 | **3.85** |
| 128 | 32 | 0.94 | **3.99** | 3.36 | 3.82 |
| 64 | 64 | 0.48 | 3.32 | **3.49** | 2.82 |
| 32 | 128 | 0.47 | 3.10 | **3.43** | 2.83 |

按行的串行前缀积，顺序受约束，读加写 1 GB。striped 在这里不可用：

| 线程数 | $V$ | blocked | blocked + 向量化 | staged |
| --- | --- | --- | --- | --- |
| 512 | 8 | 0.92 | **4.17** | 2.47 |
| 256 | 16 | 0.42 | **3.85** | 1.63 |
| 128 | 32 | 0.34 | **2.76** | 0.89 |
| 64 | 64 | 0.26 | **1.80** | 0.46 |
| 32 | 128 | 0.27 | **1.60** | 0.46 |

### 取舍

**默认用向量化的 blocked。** 它在上面两组、五个 $V$ 上都是最快或与最快持平。`T.vectorized` 把每线程那一段读成 16 字节的向量访问，于是每线程仍持有连续一段，而 32 个线程的段首尾相接铺满 512 字节，两个要求同时满足，且不需要 shared memory 与同步。

**顺序不受约束且 $V$ 偏大时，striped 略优。** $V = 64$ 与 128 上它是 3.49 与 3.43，向量化是 3.32 与 3.10。向量化在大 $V$ 上要占掉每线程 $V/2$ 个寄存器（$V = 128$ 时 64 个），占用率随之下降；striped 每线程只用一个累加器。

**blocked 只在 $V$ 很小时可以接受。** $V = 8$ 时它是 3.02，与另外三种同速 —— 冗余被 L1 吸收了。$V$ 再大就不成立：$V = 64$ 时降到 0.48。

**staged 在这两组实测里从未胜出。** 顺序受约束时它比向量化慢 1.7 到 3.9 倍，$V$ 越大差距越大 —— 一次 shared memory 中转、两次同步，以及消费阶段的 bank 冲突（见[第 2 条](#bank-conflict)）都要计入。它的用处在别处：整行需要被 block 内所有线程共享，而不是各线程只读自己那一段时。

### 代码

反例，逐元素的 blocked：

```python
for c in T.serial(V):
    acc[0] = acc[0] * T.cast(x[row, tx * V + c], "float32")
```

正例，向量化的 blocked：

```python
buf = T.alloc_local((V,), dtype)

# 一条 16 字节向量访问，生成的 CUDA 里是 *(uint4*)(buf) = *(uint4*)(x + ...)
for c in T.vectorized(V):
    buf[c] = x[row, tx * V + c]

for c in T.serial(V):
    acc[0] = acc[0] * T.cast(buf[c], "float32")
```

正例，$V$ 偏大且顺序不受约束时的 striped：

```python
for c in T.serial(V):
    acc[0] = acc[0] * T.cast(x[row, c * threads + tx], "float32")
```

## 2. 消除 shared memory 的 bank 冲突 {#bank-conflict}

[第 1 条](#coalescing)的 staged 路线把整行搬进 shared memory，每个线程再读自己那一段。消费时固定段内偏移，一个 warp 的 32 个线程相隔一整段 —— 这个 stride 决定它们落在几条 bank 上。

设一段有 `chunk` 个元素、元素 `E` 字节，则 stride 折成 4 字节 word 是 `chunk * E / 4` 个。落到多少条不同的 bank 上由它与 32 的最大公约数决定：

```
不同的 bank 数 = 32 / gcd(stride_words, 32)
冲突路数       = gcd(stride_words, 32)
```

`gcd` 等于 1 时 32 个线程铺在 32 条 bank 上，无冲突；等于 32 时全部挤在同一条 bank 上，32 路串行。fp16、`chunk = 64` 就是后者：stride 是 128 字节即 32 个 word，`gcd(32, 32) = 32`。

在 stride 上加 `pad` 个元素即可改变这个公约数。**目标是让 `(chunk + pad) * E / 4` 成为奇数**，此时 `gcd` 为 1，冲突消失。

### 实测

H200，SM 时钟锁在 1830 MHz。fp16，每行 4096 个元素，行数取 65536 使输入为 512 MB（大于 60 MiB 的 L2）。kernel 把整行搬进 `(threads, chunk + pad)` 的 shared 缓冲，逐段做串行前缀积再写回，读加写共 1 GB。括号内是上面公式预测的冲突路数：

| chunk | 线程数 | pad = 0 | pad = 2 | pad = 4 | pad = 8 | pad = 16 |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 256 | 1.63（8 路） | 2.99（1 路） | **3.29**（2 路） | 2.58（4 路） | 0.86（16 路） |
| 32 | 128 | 0.89（16 路） | 3.32（1 路） | **3.35**（2 路） | 2.57（4 路） | 1.54（8 路） |
| 64 | 64 | 0.46（32 路） | 3.63（1 路） | **3.70**（2 路） | 2.74（4 路） | 1.62（8 路） |
| 128 | 32 | 0.46（32 路） | 3.25（1 路） | **3.31**（2 路） | 2.76（4 路） | 1.61（8 路） |

20 个配置里，带宽随预测的路数单调下降 —— 1 路与 2 路在 3.0 以上，4 路降到 2.6 附近，8 路及以上跌到 1.6 以下。公式可以直接用来选 pad，不必逐个试。

### 取舍

**按公式选 pad，不要按字节对齐选。** fp16 下 `pad = 2`（4 字节）使 `(chunk + 2) * 2 / 4 = (chunk + 2) / 2` 为奇数，无冲突；`pad = 8`（16 字节）反而落在 4 路，`pad = 16`（32 字节）落在 8 路甚至 16 路。16 字节对齐能保住 shared memory 一侧的向量访问，但代价是引入的冲突远大于收益：`pad = 8` 的 2.57 到 2.76 明显低于 `pad = 2` 的 3.25 到 3.63。

**pad 之后最优的 chunk 会变。** `pad = 0` 时最优是 `chunk = 16`（1.63），`pad = 4` 时最优是 `chunk = 64`（3.70）。冲突消掉之后，更大的段意味着更少线程争用同一批 bank，最优点因此往大移。改完 pad 要重扫 chunk。

**这一条只在 staged 路线上出现。** 第 1 条的实测表明，向量化的 blocked 通常更快，那条路线把数据放进寄存器，不经 shared memory，也就没有 bank 冲突可言。只有整行需要被 block 内所有线程共享时才走 staged，届时这一条适用。

### 代码

反例，stride 恰好是 32 个 word 的倍数：

```python
staged = T.alloc_shared((threads, chunk), dtype)   # stride = chunk
```

正例，pad 到 stride 的 word 数为奇数：

```python
# fp16：chunk 为偶数时 pad = 2 即可让 (chunk + 2) * 2 / 4 为奇数
pad = 2 if elem_bytes == 2 else 1
staged = T.alloc_shared((threads, chunk + pad), dtype)
```

## 3. 用 T.copy 把整行读进 fragment {#registers}

一个线程要看过整行（或整行的一段）时，有两种写法：逐个元素从 global memory 取，或者一次 `T.copy` 把整行搬进 fragment 再遍历。

fragment 落在寄存器里，`T.copy` 的线程映射由布局推导给出、是合并的 —— 也就是[第 1 条](#coalescing)里的向量化路线，只是粒度从「每线程一段」放大到「整行」。附带的好处是列下标就是遍历 fragment 的循环变量，不必再从线程号和迭代次数拼出来。

### 实测

H200，SM 时钟锁在 1830 MHz，fp16，输入 512 MB（大于 60 MiB 的 L2）。kernel 求每行的最大值：反例逐元素走 global memory，正例先 `T.copy` 进 fragment。行宽固定、只变 block 的线程数，因此每线程持有的元素数是唯一变量：

| 行宽 | 线程数 | 每线程元素 | 每线程寄存器 | 逐元素 | `T.copy` 进 fragment |
| --- | --- | --- | --- | --- | --- |
| 4096 | 512 | 8 | 4 | 3.11 | 3.17 |
| 4096 | 256 | 16 | 8 | 3.38 | **4.24** |
| 4096 | 128 | 32 | 16 | 3.70 | **4.38** |
| 4096 | 64 | 64 | 32 | 3.97 | **4.35** |
| 4096 | 32 | 128 | 64 | 3.83 | **4.33** |
| 16384 | 128 | 128 | 64 | 3.69 | **4.44** |
| 16384 | 64 | 256 | 128 | 3.74 | **4.39** |

fragment 一侧在 8 到 128 个寄存器之间基本平坦，`4.24` 到 `4.44` TB/s；只在 512 线程那一行退回与逐元素同速。

### 判断寄存器是否够用

每线程 255 个 32 位寄存器是架构上限，超过之后编译器把放不下的部分挪到 local memory。但**这件事本身不一定有代价**：

| 每线程元素 | 每线程寄存器 | 遍历 fragment 1 遍 | 遍历 4 遍 |
| --- | --- | --- | --- |
| 16 | 8 | 4.04 | 3.78 |
| 64 | 32 | 4.20 | 4.01 |
| 256 | 128 | 4.25 | 3.73 |
| 512 | **256** | 4.05 | **2.62** |

每线程 512 个 fp16 元素折合 256 个寄存器，已经越过上限，生成的 CUDA 里确实是一个 `[512]` 的数组；只遍历一遍时带宽仍有 4.05 TB/s，遍历四遍才掉到 2.62。原因是 local memory 的访问本身是合并的、且经 L1 与 L2 缓存，遍历一遍时它的延迟被 DRAM 侧掩掉；反复遍历才把它暴露成瓶颈。

### 取舍

**优先 `T.copy` 进 fragment。** 除了 512 线程那一档，它在所有配置上都比逐元素快 0.5 到 1.0 TB/s。

**不必为了「装得进寄存器」去压线程数。** 上面 8 到 128 个寄存器的区间里带宽是平的，甚至越过 255 上限也只在 fragment 被反复遍历时才掉。线程数取 64 到 256 之间都可以，512 明显偏大。

**fragment 要被反复遍历时，才需要核对寄存器数量。** 判断依据是「每线程持有的 32 位量 = 元素数 × 元素字节 / 4」，越过 255 且要多次遍历时，改成分块处理或退回 shared memory（[第 2 条](#bank-conflict)）。

### 代码

反例，逐元素走 global memory：

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

## 4. 输出比输入窄时，每线程多取一次输入 {#narrow-output}

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

## 5. kernel 产生 bool 时，用 int8 存储 {#bool-storage}

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
