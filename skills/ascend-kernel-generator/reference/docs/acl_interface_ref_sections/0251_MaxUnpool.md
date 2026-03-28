### MaxUnpool

## 功能

MaxUnpool 本质上计算了 MaxPool 操作的部分逆运算。此操作的输入信息通常是 MaxPool 操作的输出信息。

- 第一个输入 Tensor `x` 是需要解除池化的张量，通常是 MaxPool 的第一个输出（池化张量）。
- 第二个输入 Tensor `i` 包含对应于第一个输入 Tensor `x` 中元素的（局部最大）元素的索引。输入 Tensor `i` 通常是 MaxPool 操作的第二个输出。
- 第三个（可选）输入是一个 Tensor，指定解除池化操作的输出大小。

## 输入

- **x**：输入 Tensor，数据类型支持 float16、float。
- **i**：输入 Tensor，数据类型支持 int64。
- **output_shape**：（可选），设置输出的 shape，数据类型：int64。

## 属性

- **kernel_shape**（必选）：一个列表，数据类型为 INTS，支持 int 或 int 列表，沿每个轴的内核大小。
- **pads**：一个列表，数据类型为 INTS，支持 int 或 int 列表，沿每个轴 pad。
- **strides**：一个列表，数据类型为 INTS，支持 int 或 int 列表，沿每个轴步长。

## 输出

- **y**：输出 Tensor，数据类型和输入 `x` 一致。

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v9/v11/v12/v13/v14/v15/v16/v17/v18
