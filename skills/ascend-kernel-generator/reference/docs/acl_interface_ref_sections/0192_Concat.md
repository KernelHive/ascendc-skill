### Concat

## 功能

对多个 Tensor 进行拼接，将张量列表连接成一个张量。所有输入张量必须具有相同的形状，但要连接的轴的维度大小除外。

## 输入

- **inputs**：用于串联的多个 Tensor 的列表，多个输入数据类型必须保持一致。支持的数据类型包括：
  - float16
  - float
  - int8
  - int16
  - int32
  - int64
  - uint8
  - uint16
  - uint32
  - uint64

## 属性

- **axis**（可选）：数据类型为 int，指定进行 concat 操作的轴。负值表示从后往前对维度计数，取值范围为 `[-r, r - 1]`，其中 `r` 是输入 `inputs` 的维数。

## 输出

- **concat_result**：输出级联 Tensor，数据类型与输入数据一致。

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
