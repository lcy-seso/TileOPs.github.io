# 性能指南

分三类：已经测出来的数字、自己动手定位问题的工具，以及定位之后怎么改。

| 文档 | 内容 |
| --- | --- |
| [性能数据](../benchmarks/index.md) | 每晚在 H200 上的实测，逐算子逐 workload 与最快的其他实现对比 |
| [核内时间线追踪](trace-timeline.md) | 在 kernel 体内加标记，读回逐 CTA 的时间线：空隙、停等、生产者与消费者的重叠 |
| [优化 access pattern](memory-bound-kernels.md) | 四条：访存合并、bank 冲突、整行读进 fragment、让输入与输出都用满向量访问 |
| [超越函数的开销与精度](transcendental.md) | 两条：在 fp32 上做运算、用恒等式减少超越函数的个数 |
| [TileLang 使用陷阱](tilelang-pitfalls.md) | 三条：宏按提及展开、简化器消掉 `x + 0.0`、简化器把 `x != x` 判 NaN 折成恒假 |

一次调用的计算量与访存量由 `op.eval_roofline()` 给出，取自 manifest 的 `roofline` 字段，是读测量结果时对照的上限；模型与字段规范见 [Roofline](../design/roofline.md)。
