### Expand

## 功能
将输入 Tensor 广播到指定 shape。

## 输入
- **x**：输入 Tensor
  - 数据类型：float16、float、bool、int64、int32、int8、uint8
  - 数据格式：ND
- **shape**：输入 Tensor
  - 数据类型：int64

## 输出
- **y**：输出 Tensor
  - 数据类型：与 x 保持一致
  - shape：根据输入 shape 推导得出
  - 数据格式：ND

## 约束
需要修改模型将输入 shape 由 placeholder 改为 const 类型，可以使用 onnxsimplifier 简化模型。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
