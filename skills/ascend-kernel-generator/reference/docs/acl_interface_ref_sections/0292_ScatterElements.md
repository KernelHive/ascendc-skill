### ScatterElements

## 功能

根据 `updates` 和 `indices` 来更新 `data` 的值，并把结果返回。与 `Scatter` 不同的是它提供了更精细的控制，允许指定要更新的元素在维度上的具体位置。

## 输入

- **data**：输入 Tensor，数据类型：float16、float、int32
- **indices**：Tensor，数据类型：int32、int64，shape 维度与输入 `data` 一致
- **updates**：Tensor，数据类型：float16、float、int32，shape 与 `indices` 维度一致

## 输出

- **y**：Tensor，和输入 `data` 的 shape、数据类型一致

## 属性

- **axis**：int，默认是 0，表示沿 axis 取数据

## 支持的 ONNX 版本

Opset v11/v12/v13/v14/v15/v16/v17/v18
