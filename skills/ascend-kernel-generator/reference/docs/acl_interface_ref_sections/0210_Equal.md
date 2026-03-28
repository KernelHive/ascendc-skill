### Equal

## 功能
计算两个输入 Tensor 是否有相同的大小和元素，返回一个 Bool 类型。

## 输入
- **A**：输入 Tensor
  - 数据类型：float16、float、double、int32、int8、uint8、bool、int64、int16、uint16、uint32、uint64
  - 数据格式：ND
- **B**：输入 Tensor
  - 数据类型和 shape 与输入 A 一致

## 输出
- **y**：输出 Tensor
  - 数据类型：bool

## 约束与限制
无

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
