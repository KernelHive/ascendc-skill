### Where

## 功能
根据 condition，从 X 或 Y 中返回相应的元素。

## 输入
- **condition**：bool 类型，若为 True，返回 X；否则，返回 Y。
- **X**：输入 Tensor，数据类型支持 float16、float、uint8、int8、int32。
- **Y**：输入 Tensor，数据类型与 X 保持一致。

## 输出
- **output**：输出 Tensor，数据类型与 X 和 Y 保持一致，shape 为 X 和 Y 经过广播后的 shape。

## 限制与约束
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
