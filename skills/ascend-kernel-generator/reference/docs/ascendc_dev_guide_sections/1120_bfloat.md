###### bfloat

> 本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

## 接口列表

| 接口定义 | 功能说明 |
|----------|----------|
| `from_bits()` | 空结构体，暂时没用。 |
| `from_bits_t()` | 空结构体，暂时没用。 |
| `bfloat16(uint16_t bits, [[maybe_unused]] from_bits_t fromBits)` | bfloat16的构造函数。 |
| `bfloat16()` |  |
| `bfloat16(float v)` |  |
| `bfloat16(const complex64 &val)` |  |
| `bfloat16(const complex128 &val)` |  |
| `bfloat16(const T &val)` |  |
| `float()` | 将结构体struct bfloat16成员变量value从uint16_t型转为float型。 |
| `bool()` | 判断当前bfloat16数值的绝对值是否大于float数据类型的epsilon值。 |
| `epsilon()` | 返回bfloat16数据类型的epsilon值。 |
| `short()` | 将当前bfloat16数值转换为short类型。 |
| `int()` | 将当前bfloat16数值转换为int类型。 |
| `long()` | 将当前bfloat16数值转换为long类型。 |
| `char()` | 将当前bfloat16数值转换为char类型。 |
| `double()` | 将当前bfloat16数值转换为double类型。 |
| `complex64()` | 将当前bfloat16数值转换为complex64类型。 |
| `complex128()` | 将当前bfloat16数值转换为complex128类型。 |
| `round_to_bfloat16(float v)` | 将浮点数转换为bfloat16，舍入方法为round-nearest-to-even。 |
| `highest()` | bfloat16数据类型的最大值。 |
| `lowest()` | bfloat16数据类型的最小值。 |
| `min_positive_normal()` | bfloat16数据类型最小的正数正常值。 |
| `IsZero()` | 判断当前bfloat16数值是不是0。 |
| `float_isnan(const float &x)` | 判断float类型数值x是不是非正常数。 |
| `isinf(const bfloat16 &a)` | 判断给定的bfloat16数值a是不是无穷大。 |
| `isnan(const bfloat16 &a)` | 判断给定的bfloat16数值a是否不是一个数。 |
| `isfinite(const bfloat16 &a)` | 判断给定的bfloat16数值a是否为有限值。 |
| `exp(const bfloat16 &a)` | 计算e的a次幂。 |
| `log(const bfloat16 &a)` | 计算a的自然对数。 |
| `log10(const bfloat16 &a)` | 计算以10为底的a的对数。 |
| `sqrt(const bfloat16 &a)` | 计算a的平方根。 |
| `pow(const bfloat16 &a, const bfloat16 &b)` | 计算a的b次幂。 |
| `sin(const bfloat16 &a)` | 计算a角度的正弦值。 |
| `cos(const bfloat16 &a)` | 计算a角度的余弦值。 |
| `tan(const bfloat16 &a)` | 计算a角度的正切值。 |
| `tanh(const bfloat16 &a)` | 计算a角度的双曲正切值。 |
| `floor(const bfloat16 &a)` | 返回不大于bfloat16数值a的最大整数值。 |
| `ceil(const bfloat16 &a)` | 返回不小于bfloat16数值a的最小整数值。 |
| `min()` | 返回bfloat16数据类型最小的正数正常值。 |
| `max()` | 返回bfloat16数据类型的最大值。 |
| `round_error()` | 最大舍入误差的量。 |
| `infinity()` | 返回bfloat16数据类型的无穷大值。 |
| `quiet_NaN()` | 返回bfloat16数据类型的quite(non-signaling) NaN值。 |
| `signaling_NaN()` | 返回bfloat16数据类型的signaling NaN值。 |
| `denorm_min()` | 返回bfloat16数据类型最小正非规格化值。 |
