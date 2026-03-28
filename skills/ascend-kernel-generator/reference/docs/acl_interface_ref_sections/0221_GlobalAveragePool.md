### GlobalAveragePool

## 功能

对输入 Tensor 在同一通道中的值上应用平均池化。

## 输入

- **x**：输入 Tensor
  - 格式：ND、NCHW、NCDHW
  - 维度：最小 4 维，最大不超过 8 维
  - 数据类型：float16、float

## 输出

- **y**：输出 Tensor
  - 数据类型和格式与输入 x 相同
  - 维度与 x 相同，前两维与 x 一致，后续维度为 1

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
