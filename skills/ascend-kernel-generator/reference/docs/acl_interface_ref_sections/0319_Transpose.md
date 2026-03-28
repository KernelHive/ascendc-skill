### Transpose

## 功能
对输入 Tensor 进行转置。

## 输入
- **data**：输入 Tensor，数据类型支持 float16、float、int8、int16、int32、int64、uint8、uint16、uint32、uint64、bool。

## 属性
- **perm**：int 列表，用于指示输入 Tensor 转置后每个维度的位置。

## 输出
- **transposed**：输出 Tensor，数据类型与 `data` 保持一致。

## 限制与约束
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
