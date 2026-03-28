### SpaceToDepth

## 功能

将输入 Tensor 中空间维度的数据重新排列到深度维度，即将高度和宽度维度的数据移动到深度维度。

## 输入

- **input**：输入 Tensor，数据类型支持 float16、float，shape 为 `[N, C, H, W]`，其中：
  - N 为 batch
  - C 为 channel 或 depth
  - H 为 height
  - W 为 width

## 属性

- **blocksize**：int 类型整数，用于指定分别在高度和深度维度移动的数据量。

## 输出

- **output**：输出 Tensor，数据类型与 input 保持一致，shape 为 `[N, C * blocksize * blocksize, H / blocksize, W / blocksize]`。

## 限制与约束

无。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
