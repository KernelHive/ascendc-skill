###### VectorPadding(ISASI)

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

根据 `padMode`（pad模式）与 `padSide`（pad方向）对源操作数按照 datablock 进行填充操作。

假设源操作数的一个 datablock 有 16 个数，`datablock[0:15] = a~p`：

- `padSide == false`：从 datablock 的左边开始填充，即 datablock 的起始值方向（a→p）
- `padSide == true`：从 datablock 的右边开始填充，即 datablock 的结束值方向（p→a）
- `padMode == 0`：用邻近数作为填充值，例：`aaa|abc`（`padSide=false`）、`nop|ppp`（`padSide=true`）
- `padMode == 1`：用邻近 datablock 值对称填充，例：`cba|abc`（`padSide=false`）、`nop|pon`（`padSide=true`）
- `padMode == 2`：用邻近 datablock 值填充，偏移一个数，做对称填充，例：
  - `padSide = false`：`xcb|abc`，xcb 被填充，填充过程描述：a 被丢弃，对称填充，x 处填充 0
  - `padSide = true`：`nop|onx`，onx 被填充，填充过程描述：p 被丢弃，对称填充，x 处填充 0

## 函数原型

### tensor 前 n 个数据计算

```cpp
template <typename T>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint32_t count)
```

### tensor 高维切分计算

#### mask 逐 bit 模式

```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint64_t mask[], const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

#### mask 连续模式

```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint8_t padMode, const bool padSide, const uint64_t mask, const uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

## 参数说明

### 表 15-97 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数数据类型。<br>Atlas 推理系列产品 AI Core，支持的数据类型为：`int16_t`/`uint16_t`/`half`/`int32_t`/`uint32_t`/`float` |
| isSetMask | 是否在接口内部设置 mask。<br>• `true`：表示在接口内部设置 mask。<br>• `false`：表示在接口外部设置 mask，开发者需要使用 `SetVectorMask` 接口设置 mask 值。这种模式下，本接口入参中的 mask 值必须设置为占位符 `MASK_PLACEHOLDER`。 |

### 表 15-98 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dst | 输出 | 目的操作数。<br>类型为 `LocalTensor`，支持的 TPosition 为 `VECIN`/`VECCALC`/`VECOUT`。<br>`LocalTensor` 的起始地址需要 32 字节对齐。 |
| src | 输入 | 源操作数。<br>类型为 `LocalTensor`，支持的 TPosition 为 `VECIN`/`VECCALC`/`VECOUT`。<br>`LocalTensor` 的起始地址需要 32 字节对齐。<br>源操作数的数据类型需要与目的操作数保持一致。 |
| padMode | 输入 | padding 模式，类型为 `uint8_t`，取值范围：[0,2]。<br>• 0：用邻近数作为填充值。<br>• 1：用邻近 datablock 值对称填充。<br>• 2：用邻近 datablock 值填充，偏移一个数，做对称填充。 |
| padSide | 输入 | padding 的方向，类型为 `bool`。<br>• `false`：左边。<br>• `true`：右边。 |
| count | 输入 | 参与计算的元素个数。 |
| mask[]/mask | 输入 | mask 用于控制每次迭代内参与计算的元素。<br>• **逐 bit 模式**：可以按位控制哪些元素参与计算，bit 位的值为 1 表示参与计算，0 表示不参与。<br>mask 为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为 16 位时，数组长度为 2，`mask[0]`、`mask[1]` ∈ [0, 2^64-1] 并且不同时为 0；当操作数为 32 位时，数组长度为 1，`mask[0]` ∈ (0, 2^64-1]；当操作数为 64 位时，数组长度为 1，`mask[0]` ∈ (0, 2^32-1]。<br>例如，`mask = [8, 0]`，`8 = 0b1000`，表示仅第 4 个元素参与计算。<br>• **连续模式**：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为 16 位时，`mask` ∈ [1, 128]；当操作数为 32 位时，`mask` ∈ [1, 64]；当操作数为 64 位时，`mask` ∈ [1, 32]。 |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的 256 Bytes 数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。<br>`repeatTime` 表示迭代的次数。<br>关于该参数的具体描述请参考 12.3 如何使用 Tensor 高维切分计算 API。 |
| repeatParams | 输入 | 控制操作数地址步长的参数。<br>`UnaryRepeatParams` 类型，包含操作数相邻迭代间相同 DataBlock 的地址步长，操作数同一迭代内不同 DataBlock 的地址步长等参数。<br>相邻迭代间的地址步长参数说明请参考 `repeatStride`；同一迭代内 DataBlock 的地址步长参数说明请参考 `dataBlockStride`。 |

## 返回值说明

无

## 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- mask 仅控制目的操作数中的哪些元素要写入，源操作数的读取与 mask 无关。
- `count` 表示写入目的操作数中的元素总数，源操作数的读取与 `count` 无关。

## 调用示例

样例的 `srcLocal` 和 `dstLocal` 均为 `half` 类型。

更多样例可参考 LINK。

### tensor 高维切分计算样例 - mask 连续模式

```cpp
uint64_t mask = 256 / sizeof(half);
uint8_t padMode = 0;
bool padSide = false;
// repeatTime = 4, 128 elements one repeat, 512 elements total
// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::VectorPadding(dstLocal, srcLocal, padMode, padSide, mask, 4, { 1, 1, 8, 8 });
```

### tensor 高维切分计算样例 - mask 逐 bit 模式

```cpp
uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
uint8_t padMode = 0;
bool padSide = false;
// repeatTime = 4, 128 elements one repeat, 512 elements total
// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::VectorPadding(dstLocal, srcLocal, padMode, padSide, mask, 4, { 1, 1, 8, 8 });
```

### tensor 前 n 个数据计算样例

```cpp
uint8_t padMode = 0;
bool padSide = false;
AscendC::VectorPadding(dstLocal, srcLocal, padMode, padSide, 512);
```

## 结果示例

以 `srcLocal` 的一个 datablock 的值为例，有 16 个数：

```
输入数据 (srcLocal): [6.938 -8.86 -0.2263 ... 1.971 1.778]
输出数据 (dstLocal): [6.938 6.938 6.938 ... 6.938 6.938]
```
