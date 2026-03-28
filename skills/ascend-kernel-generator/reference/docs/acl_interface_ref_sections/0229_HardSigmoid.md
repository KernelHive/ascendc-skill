### HardSigmoid

## 功能

HardSigmoid 接受一个输入数据（Tensor）并生成一个输出数据（Tensor），对输入 Tensor 逐元素计算。

## 输入

- **x**：输入 Tensor，数据类型支持 float16、float。

## 属性

- **alpha**：数据类型为 float，默认值：0.2
- **beta**：数据类型为 float，默认值：0.5

## 输出

- **y**：输出 Tensor，数据类型和 shape 与输入一致。

## 约束

无。

## 支持的 ONNX 版本

Opset v1/v6/v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
