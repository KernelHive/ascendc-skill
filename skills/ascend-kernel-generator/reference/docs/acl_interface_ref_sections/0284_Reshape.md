### Reshape

## 功能
改变输入 Tensor 的形状。

## 输入
- **data**：输入 data Tensor，数据类型：float16、int8、int16、uint16、uint8、int32、int64、uint32、uint64、bool、double。
- **shape**：输入 shape Tensor，指定了输出 Tensor 的形状，数据类型：int64。

## 输出
**reshaped data**：输出 Tensor，数据类型与输入 data 一致。

## 属性
**allowzero**：数据类型 int，默认为 0。此时输入 shape 中的任何值等于零时，相应的维度值将从输入 data 中动态复制。allowzero=1 表示如果输入 shape 中的任何值为零，则遵循零值，类似于 NumPy。

## 限制与约束
无

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
