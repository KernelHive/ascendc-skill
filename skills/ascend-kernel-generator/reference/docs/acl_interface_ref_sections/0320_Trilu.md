### Trilu

## 功能

获取一个二维矩阵或一批二维矩阵的上三角或下三角部分。输入 Tensor 的 shape 为 `[*, N, M]`，其中 `*` 表示零个或多个维度。

## 输入

- **input**：输入 Tensor，数据类型支持 uint8、int8、int16、int32、bool、float、float16、float64、int64，shape 为 `[*, N, M]`。
- **k**（可选）：用于指定主对角线的偏移量，仅支持 `k=0`。

## 属性

- **upper**：int，用于指定保留矩阵的上三角还是下三角部分：
  - 若为 1，保留上三角部分；
  - 若为 0，保留下三角部分。

## 输出

- **output**：输出 Tensor，数据类型和 shape 与 input 保持一致。

## 限制与约束

无。

## 支持的 ONNX 版本

Opset v14/v15/v16/v17/v18
