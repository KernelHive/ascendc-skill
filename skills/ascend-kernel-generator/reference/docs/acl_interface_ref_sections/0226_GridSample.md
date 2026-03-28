### GridSample

## 功能

给定一个输入 Tensor `x` 和一个输入 Tensor `grid`（网格），然后根据 `grid` 中每个位置提供的坐标信息，将 `x` 中对应位置的值填充到网格指定位置，得到最终输出 Tensor `y`。

## 输入

- **x**：四维 Tensor，形状为 `[batch, channels, height_in, width_in]`，支持非连续 Tensor，数据格式支持 ND，数据类型：float16、float32。
- **grid**：四维 Tensor，形状为 `[batch, height_out, width_out, 2]`，支持非连续 Tensor，数据格式支持 ND，数据类型：float16、float32。

## 属性

- **interpolation_mode**：数据类型 string，指定插值方法的可选字符串：
  - `0`：bilinear（双线性插值）
  - `1`：nearest（最邻近插值）
  - `2`：bicubic（双三次插值）

  目前只支持 bilinear 和 nearest，默认是 `0`。

- **padding_mode**：数据类型 string，指定填充模式，即当 (x, y) 取值超过输入特征图采样范围时，返回一个特定值：
  - `0`：zeros
  - `1`：border
  - `2`：reflection

- **align_corners**：数据类型 bool，默认为 `false`。如果为 `true`，则输入和输出 Tensor 的角点像素的中心对齐。

## 输出

**y**：四维 Tensor，形状为 `[batch, channels, height_out, width_out]`，且数据类型与 `x` 的数据类型一致，支持非连续 Tensor，数据格式支持 ND，数据类型：float16、float32。

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v16
