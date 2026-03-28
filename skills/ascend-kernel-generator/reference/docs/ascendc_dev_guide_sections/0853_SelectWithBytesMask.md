##### SelectWithBytesMask

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

给定两个源操作数 `src0` 和 `src1`，根据 `maskTensor` 相应位置的值（非bit位）选取元素，得到目的操作数 `dst`。选择的规则为：当 Mask 的值为 0 时，从 `src0` 中选取，否则从 `src1` 选取。

该接口支持多维 Shape，需满足 `maskTensor` 和源操作数 Tensor 的前轴（非尾轴）元素个数相同，且 `maskTensor` 尾轴元素个数大于等于源操作数尾轴元素个数，`maskTensor` 多余部分丢弃不参与计算。

- `maskTensor` 尾轴需 32 字节对齐且元素个数为 16 的倍数
- 源操作数 Tensor 尾轴需 32 字节对齐

如下图样例，源操作数 `src0` 为 Tensor，shape 为 (2,16)，数据类型为 half，尾轴长度满足 32 字节对齐；源操作数 `src1` 为 scalar，数据类型为 half；`maskTensor` 的数据类型为 bool，为满足对齐要求 shape 为 (2,32)，仅有图中蓝色部分的 mask 掩码生效，灰色部分不参与计算。输出目的操作数 `dstTensor` 如下图所示。

## 实现原理

以 float 类型，ND 格式，shape 为 [m, k1] 的 source 输入 Tensor，shape 为 [m, k2] 的 mask Tensor 为例，描述 SelectWithBytesMask 高阶 API 内部算法框图，如下图所示。

**图 15-82 SelectWithBytesMask 算法框图**

计算过程分为如下几步，均在 Vector 上进行：

1. **GatherMask 步骤**：如果 k1, k2 不相等，则根据 src 的 shape [m, k1]，对输入 mask [m, k2] 通过 GatherMask 进行 reduce 计算，使得 mask 的 k 轴多余部分被舍去，shape 转换为 [m, k1]
2. **Cast 步骤**：将上一步的 mask 结果 cast 成 half 类型
3. **Compare 步骤**：使用 Compare 接口将上一步的 mask 结果与 0 进行比较，得到 `cmpmask` 结果
4. **Select 步骤**：根据 `cmpmask` 的结果，选择 `srcTensor` 相应位置的值或者 scalar 值，输出 Output

## 函数原型

### 原型 1：src0 为 srcTensor（tensor类型），src1 为 srcScalar（scalar类型）

```cpp
template <typename T, typename U, bool isReuseMask = true>
__aicore__ inline void SelectWithBytesMask(
    const LocalTensor<T> &dst,
    const LocalTensor<T> &src0,
    T src1,
    const LocalTensor<U> &mask,
    const LocalTensor<uint8_t> &sharedTmpBuffer,
    const SelectWithBytesMaskShapeInfo &info
)
```

### 原型 2：src0 为 srcScalar（scalar类型），src1 为 srcTensor（tensor类型）

```cpp
template <typename T, typename U, bool isReuseMask = true>
__aicore__ inline void SelectWithBytesMask(
    const LocalTensor<T> &dst,
    T src0,
    const LocalTensor<T> &src1,
    const LocalTensor<U> &mask,
    const LocalTensor<uint8_t> &sharedTmpBuffer,
    const SelectWithBytesMaskShapeInfo &info
)
```

该接口需要额外的临时空间来存储计算过程中的中间变量。临时空间需要开发者申请并通过 `sharedTmpBuffer` 入参传入。临时空间大小 `BufferSize` 的获取方式如下：通过 15.1.5.10.2 `GetSelectWithBytesMaskMaxMinTmpSize` 中提供的接口获取需要预留空间范围的大小。

## 参数说明

### 表 15-856 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float<br>Atlas 推理系列产品AI Core，支持的数据类型为：half/float |
| U | 掩码 Tensor mask 的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：bool/uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：bool/uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t<br>Atlas 推理系列产品AI Core，支持的数据类型为：bool/uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t |
| isReuseMask | 是否允许修改 maskTensor。默认为 true。<br>取值为 true 时，仅在 maskTensor 尾轴元素个数和 srcTensor 尾轴元素个数不同的情况下，maskTensor 可能会被修改；其余场景，maskTensor 不会修改。<br>取值为 false 时，任意场景下，maskTensor 均不会修改，但可能会需要更多的临时空间。 |

### 表 15-857 接口参数说明

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| dst | 输出 | 目的操作数。<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT。 |
| src0(srcTensor) | 输入 | 源操作数。源操作数 Tensor 尾轴需 32 字节对齐。<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT。 |
| src1(srcTensor) | 输入 | 源操作数。源操作数 Tensor 尾轴需 32 字节对齐。<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT。 |
| src1(srcScalar) | 输入 | 源操作数。类型为 scalar。 |
| src0(srcScalar) | 输入 | 源操作数。类型为 scalar。 |
| mask | 输入 | 掩码 Tensor。用于描述如何选择 srcTensor 和 srcScalar 之间的值。maskTensor 尾轴需 32 字节对齐且元素个数为 16 的倍数。<br>• src0 为 srcTensor（tensor类型），src1 为 srcScalar（scalar类型）：若 mask 的值为 0，选择 srcTensor 相应的值放入 dstLocal，否则选择 srcScalar 的值放入 dstLocal。<br>• src0 为 srcScalar（scalar类型），src1 为 srcTensor（tensor类型）：若 mask 的值为 0，选择 srcScalar 的值放入 dstLocal，否则选择 srcTensor 相应的值放入 dstLocal。 |
| sharedTmpBuffer | 输入 | 该 API 用于计算的临时空间，所需空间大小根据 15.1.5.10.2 GetSelectWithBytesMaskMaxMinTmpSize 获取。 |
| info | 输入 | 描述 SrcTensor 和 maskTensor 的 shape 信息。<br>`SelectWithBytesMaskShapeInfo` 类型，定义如下：<br>```cpp<br>struct SelectWithBytesMaskShapeInfo {<br>    __aicore__ SelectWithBytesMaskShapeInfo(){};<br>    uint32_t firstAxis = 0;<br>    uint32_t srcLastAxis = 0;<br>    uint32_t maskLastAxis = 0;<br>};<br>```<br>• `firstAxis`：srcLocal/maskTensor 的前轴元素个数<br>• `srcLastAxis`：srcLocal 的尾轴元素个数<br>• `maskLastAxis`：maskTensor 的尾轴元素个数<br>**注意：**<br>• 需要满足 srcTensor 和 maskTensor 的前轴元素个数相同，均为 `firstAxis`<br>• 需要满足 `firstAxis * srcLastAxis = srcTensor.GetSize()`；`firstAxis * maskLastAxis = maskTensor.GetSize()`<br>• maskTensor 尾轴的元素个数大于等于 srcTensor 尾轴的元素个数，计算时会丢弃 maskTensor 多余部分，不参与计算 |

## 返回值说明

无

## 约束说明

- 源操作数与目的操作数可以复用
- 操作数地址对齐要求请参见通用地址对齐约束
- maskTensor 尾轴元素个数和源操作数尾轴元素个数不同的情况下，maskTensor 的数据有可能被接口改写

## 调用示例

完整调用样例请参考 `SelectWithBytesMaskCustom` 算子样例。

```cpp
AscendC::SelectWithBytesMaskShapeInfo info;
srcLocal1 = inQueueX1.DeQue<srcType>();
maskLocal = maskQueue.DeQue<maskType>();
AscendC::LocalTensor<uint8_t> tmpBuffer = sharedTmpBuffer.Get<uint8_t>();
dstLocal = outQueue.AllocTensor<srcType>();
AscendC::SelectWithBytesMask(dstLocal, srcLocal1, scalar, maskLocal, tmpBuffer, info);
outQueue.EnQue<srcType>(dstLocal);
maskQueue.FreeTensor(maskLocal);
inQueueX1.FreeTensor(srcLocal1);
```

## 结果示例

**输入数据 srcLocal1:**
```
[-84.6 -24.38 30.97 -30.25 22.28 -92.56 90.44 -58.72 -86.56 5.74 6.754 -86.3 -96.7 -37.38 -81.9 46.9
-99.4 94.2 -41.78 -60.3 -14.43 78.6 8.93 -65.2 79.94 -46.88 4.516 20.03 -25.56 24.73 0.3223 21.98
-87.4 -93.9 46.22 -69.9 90.8 -24.17 -96.2 -91. 90.44 9.766 68.25 -57.78 -75.44 -8.86 -91.56 21.6
76. 82.1 -78. -23.75 92. -66.44 75. 94.9 2.62 -90.9 15.945 38.16 50.84 96.94 -59.38 44.22 ]
```

**输入数据 scalar:**
```
[35.6]
```

**输入数据 maskLocal:**
```
[False True False False True True False True True False False True False True False True
True False False False True True True True True False True False True True True True
False False True False True False True False True False True False True True True False
True False True False True False True True True False False False True False True True]
```

**输出数据 dstLocal:**
```
[-84.6 35.6 30.97 -30.25 35.6 35.6 90.44 35.6 35.6 5.74 6.754 35.6 -96.7 35.6 -81.9 35.6
35.6 94.2 -41.78 -60.3 35.6 35.6 35.6 35.6 35.6 -46.88 35.6 20.03 35.6 35.6 35.6 35.6
-87.4 -93.9 35.6 -69.9 35.6 -24.17 35.6 -91. 35.6 9.766 35.6 -57.78 35.6 35.6 35.6 21.6
35.6 82.1 35.6 -23.75 35.6 -66.44 35.6 35.6 35.6 -90.9 15.945 38.16 35.6 96.94 35.6 35.6]
```
