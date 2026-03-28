###### SetAntiQuantVector

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

在 Matmul 计算时支持 A 矩阵 half 类型输入，B 矩阵 int8 类型输入，该场景下，需要调用伪量化接口进行伪量化。调用伪量化接口后，将数据从 GM 搬出到 L1 时，会执行伪量化操作，将 B 矩阵转化为 half 类型。

本节的伪量化接口提供一个量化参数向量，该向量的 shape 为 `[1, N]`，N 值为 Matmul 矩阵计算时 M/N/K 中的 N 值。对 B 矩阵的每一列都采用该向量中对应列的伪量化系数进行伪量化。

请在 Iterate 或者 IterateAll 之前调用该接口。

## 函数原型

```cpp
__aicore__ inline void SetAntiQuantVector(
    const LocalTensor<SrcT> &offsetTensor,
    const LocalTensor<SrcT> &scaleTensor
)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| offsetTensor | 输入 | 伪量化运算时的参数向量，用于加。SrcT 为 A_TYPE 中对应的数据类型。 |
| scaleTensor | 输入 | 伪量化运算时的参数向量，用于乘。SrcT 为 A_TYPE 中对应的数据类型。 |

## 返回值说明

无

## 约束说明

无
