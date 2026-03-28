### HardSwish

## 功能

HardSwish 激活函数，接受一个输入数据（Tensor）并生成一个输出数据（Tensor）。

其中 alpha=1/6，beta=0.5，对输入 Tensor 逐元素计算。

## 输入

- **x**：输入 Tensor，数据类型支持 float16、float。

## 输出

- **y**：输出 Tensor，数据类型和 shape 与输入一致。

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v14/v15/v16/v17/v18
