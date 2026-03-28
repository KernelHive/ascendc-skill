### Softmax

## 功能
对输入 Tensor 应用 Softmax 函数。

## 输入
- **input**：输入 Tensor，数据类型支持 float16、float、double。

## 属性
- **axis**（可选）：int，用于指定进行 Softmax 计算的轴，默认值为 -1。

## 输出
- **output**：输出 Tensor，数据类型和 shape 与 input 保持一致。

## 限制与约束
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
