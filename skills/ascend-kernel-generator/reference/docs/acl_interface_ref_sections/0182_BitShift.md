### BitShift

## 功能

元素级位移算子，按位移位运算符执行，按元素运算。

## 输入

- **x**：输入 Tensor，表示被位移的输入，数据类型支持 int8、int16、int32、uint8、uint16、uint32、uint64。
- **y**：输入 Tensor，表示位移的数量，数据类型支持 int8、int16、int32、uint8、uint16、uint32、uint64。

## 属性

- **direction**：数据类型为 string，指定位移方向，取值范围为 `"RIGHT"`（用于右移）或者 `"LEFT"`（用于左移）。

## 输出

- **z**：输出 Tensor，表示位移后的结果，数据类型支持 int8、int16、int32、uint8、uint16、uint32、uint64。

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v11/v12/v13/v14/v15/v16/v17/v18
