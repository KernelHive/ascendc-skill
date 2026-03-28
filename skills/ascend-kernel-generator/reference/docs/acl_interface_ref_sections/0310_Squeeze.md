### Squeeze

## 功能
移除输入 Tensor 的 shape 中数值为 1 的维度。

## 输入
- **data**：输入 Tensor，数据类型支持 float16、float、double、uint8、uint16、uint32、uint64、int8、int16、int32、int64、bool。

## 属性
- **axes**：int 列表，用于指定需要移除的维度。数据类型支持 int32、int64。

## 输出
- **squeezed**：输出 Tensor，数据类型与 `data` 保持一致。

## 限制与约束
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
