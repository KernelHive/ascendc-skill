### Flatten

## 功能

将输入张量 `input` 展平为 2D 矩阵。如果输入张量的形状为 `(d_0, d_1, ...d_n)`，则输出的形状为 `(d_0 * d_1 ...d_axis-1, d_axis * ...* dn)`。

## 输入

- **input**：输入 Tensor，秩 r > 2，数据类型：float16、float。

## 属性

- **axis**：数据类型支持 int（默认为 1）。指定一个输入维度（不包含）平铺到输出的外部维度。axis 的值必须在 `[-r, r]` 范围内，其中 r 是输入 Tensor 的秩。负值表示从后面开始计算尺寸。当 `axis = 0` 时，输出 Tensor 的形状为 `(1, (d_0 * d_1...d_n))`，其中输入 Tensor 的形状为 `(d_0, d_1, ...d_n)`。

## 输出

- **output**：输出 Tensor，具有输入 tensor 的内容的 2D Tensor。

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
