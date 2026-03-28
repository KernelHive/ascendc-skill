###### SetAntiQuantScalar

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

在 Matmul 计算时支持 A 矩阵 half 类型输入，B 矩阵 int8 类型输入，该场景下，需要调用伪量化接口进行伪量化。调用伪量化接口后，将数据从 GM 搬出到 L1 时，会执行伪量化操作，将 B 矩阵转化为 half 类型。本节的伪量化接口提供对 B 矩阵的所有数据采用同一量化系数进行伪量化的功能。

请在 Iterate 或者 IterateAll 之前调用该接口。

## 函数原型

```cpp
__aicore__ inline void SetAntiQuantScalar(const SrcT offsetScalar, const SrcT scaleScalar)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| offsetScalar | 输入 | 伪量化系数，用于加法。SrcT 为 A_TYPE 中对应的数据类型。 |
| scaleScalar | 输入 | 伪量化系数，用于乘法。SrcT 为 A_TYPE 中对应的数据类型。 |

## 返回值说明

无

## 约束说明

无
