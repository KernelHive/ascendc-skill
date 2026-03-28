### Slice

## 功能

沿多个轴生成输入 Tensor 的切片，通过 `starts`、`ends`、`axes`、`steps` 确定每个切片的位置和大小。

## 输入

- **data**：输入 Tensor，数据类型支持 float16、float、double、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool。
- **starts**：一维 Tensor，用于指定切片的起始位置，数据类型支持 int32、int64。
- **ends**：一维 Tensor，用于指定切片的结束位置，数据类型支持 int32、int64。
- **axes（可选）**：一维 Tensor，用于指定进行切片的轴，数据类型支持 int32、int64。
- **steps（可选）**：一维 Tensor，用于指定进行切片的步长，数据类型支持 int32、int64。

## 输出

- **output**：输出 Tensor，数据类型与 `data` 保持一致。

## 限制与约束

- **data**：输入 Tensor 的维度不能为 1。
- **steps**：最后一轴的取值必须为 1。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
