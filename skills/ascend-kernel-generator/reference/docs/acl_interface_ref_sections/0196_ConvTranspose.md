### ConvTranspose

## 功能
- 转置卷积，使用一个输入张量和一个滤波器并计算输出
- 如果提供了 `pads` 参数，则通过以下公式计算输出的形状：

- `output_shape` 也可以明确指定，在这种情况下，使用以下公式自动生成 `pads`：
  - 如果 `auto_pads == SAME_UPPER`
  - 否则

## 输入
- **X**：来自上一层的输入 Tensor
  - 对于二维卷积其 shape 为 `[N, C, H, W]`，其中 N 是批量大小，C 是通道数，H 和 W 是高度和宽度
  - 对于超过 2 个维度其 shape 为 `[N, C, D, H, W]`
  - 数据类型支持 `float16`
- **W**：权重 Tensor
  - 对于二维卷积其 shape 为 `[C, M/group, kH, kW]`，其中 C 是通道数，kH 和 kW 是内核的高度和宽度，M 是特征图的数量
  - 对于超过 2 个维度权重 shape 为 `[C, M/group, kD, kH, kW]`
  - 输出中的通道数应等于 `W.shape[1] * group`（假设形状数组的索引从零开始）
  - 数据类型支持 `float16`
- **B**（可选）：输入一维 Tensor
  - shape 为 `[M]`
  - 数据类型支持 `float16`

## 属性
- **auto_pad**：数据类型为 `string`，支持 `NOTSET`、`SAME_UPPER`、`SAME_LOWER` 或 `VALID`，默认为 `NOTSET`，显式使用 padding 的方式。`SAME_UPPER` 或 `SAME_LOWER` 表示填充输入，输出 y 的 shape 有 `y_H = x_H * stride_H`，`y_W = x_W * stride_W`；`auto_pad` 设为 `VALID` 时不使用 padding
- **dilations**：数据类型为 `ints`，默认为全 1 序列，表示 filter 的每轴空洞值
- **group**：数据类型为 `int`，默认为 1，表示输入通道分组数
- **kernel_shape**：数据类型为 `ints`，默认为输入 filter 的 shape，表示卷积核大小
- **output_padding**：数据类型为 `ints`，默认为全 0 数组，表示指定 padding 值
- **output_shape**：数据类型为 `ints`，根据 pad 自动计算，表示输出 shape
- **pads**：数据类型为 `ints`，默认为全 0 矩阵，表示每根轴指定 pad 值
- **strides**：数据类型为 `ints`，默认为全 1 矩阵，表示每根轴的 stride 值

## 输出
- **y**：卷积结果的输出数据张量，数据类型和输入一致

## 约束
- 输入通道 `[C]` 需要被 `group` 整除；bias 的 shape `[M]`、filter 的第 1 维维度 `[M/group]`、输出 y 的第 1 维 `[M]` 需要满足约束
- 属性 `auto_pad` 在二维场景支持 `"SAME_UPPER"`、`"SAME_LOWER"`、`"VALID"`，三维场景不支持
- 属性 `auto_pad` 在二维场景设置为 `"SAME_UPPER"`、`"SAME_LOWER"`、`"VALID"` 时，属性 `pads` 若提供则会被修正

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
