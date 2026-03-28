### GatherElements

## 功能

获取输入 Tensor `indices` 产生的索引，确定输入 Tensor `input` 的元素生成对应输入 Tensor `indices` 位置的输出 Tensor `output`。

## 输入

- **input**：输入 Tensor，秩 r ≥ 1，数据类型：float16、float、int32。
- **indices**：输入 Tensor，索引 Tensor 与输入 `input` 具有相同的秩 r，数据类型：int32、int64。

## 属性

- **axis**：数据类型支持 int，默认为 0，指定聚集的轴，负数表示从后面计算维度。

## 输出

- **output**：输出 Tensor，与 `indices` 的 shape 相同。

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v11/v12/v13/v14/v15/v16/v17/v18
