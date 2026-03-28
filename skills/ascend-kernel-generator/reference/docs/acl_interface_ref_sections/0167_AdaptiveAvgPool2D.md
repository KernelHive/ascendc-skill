### AdaptiveAvgPool2D

## 功能
对输入进行 2D 自适应平均池化计算。

## 输入
- **x**：输入 Tensor
  - Shape: `[N, C, H_x, W_x]`
  - 数据类型：float16、float
  - 数据格式：ND

## 属性
- **output_size**：包含两个 int 的列表，指定输出 y 的 `[H_y, W_y]` 的 shape 大小。

## 输出
- **y**：输出 Tensor
  - Shape: `[N, C, H_y, W_y]`
  - 数据类型：与 x 类型一致
  - 数据格式：ND

## 约束与限制
无

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16
