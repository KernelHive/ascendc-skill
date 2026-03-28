### Split

## 功能
将输入 Tensor 沿指定轴拆分成多个 Tensor。

## 输入
- **input**：输入 Tensor，数据类型支持 float16、float、int8、int16、int32、int64、uint8、uint16、uint32、uint64。

## 属性
- **split**（Opset v13 之后作为输入）：int 列表，用于指定沿拆分轴每个输出 Tensor 的大小，数据类型支持 int8、int16、int32、int64。
- **axis**：int，用于指定拆分的轴，数据类型支持 int8、int16、int32、int64。

## 输出
- **outputs**：多个输出 Tensor，每个 Tensor 数据类型与输入 input 一致。

## 限制与约束
- **split**：列表中的每个元素须大于 1，且所有元素的和必须等于拆分维度的大小。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
