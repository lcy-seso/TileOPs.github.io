# TileLang 使用陷阱

这一页记录 TileLang 使用时的一些注意事项。它们的共同点是代码本身没有错，而生成的 CUDA 与预期不同，因此判断方法都是读 `kernel.get_kernel_source()`。

## 1. 传给 T.macro 的表达式先绑定到局部变量 {#macro-expansion}

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

## 2. 需要让 -0.0 与 +0.0 落到同一个值时，不要依赖 x + 0.0 {#signed-zero}

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
