### GlobalMaxPool

## 功能
输入一个张量，并对同一通道中的值取最大值。

## 输入
- **x**：输入Tensor
  - 支持非连续Tensor
  - 数据格式：NCHW
  - 数据类型：float16

## 输出
- **output**：输出Tensor
  - 支持非连续Tensor
  - 数据格式：NCHW
  - 数据类型：float16

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
