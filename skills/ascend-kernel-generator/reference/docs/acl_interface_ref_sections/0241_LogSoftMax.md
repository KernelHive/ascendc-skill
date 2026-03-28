### LogSoftMax

## 功能
该运算符计算给定输入 Tensor 的 softmax 值的对数。

## 输入
- **x**：输入 Tensor，数据类型支持 float16、float。

## 属性
- **axis**：数据类型为 int。指定计算的轴，取值范围：[-r, r-1]，r 为输入的秩，默认值为 -1。

## 输出
- **y**：输出 Tensor，数据类型和 shape 与输入一致。

## 约束与限制
- float16 不支持 ONNX v18。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
