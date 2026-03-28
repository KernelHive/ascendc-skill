### LayerNormalization

## 功能

层归一化函数。对指定层进行归一化计算。

## 输入

- **x**：输入 Tensor，数据类型支持 float16
- **scale**：输入 Tensor，数据类型支持 float16，指定尺度因子
- **B**（可选）：输入 Tensor，数据类型支持 float16，指定偏移量

## 属性

- **axis**：数据类型为 int，默认值为 -1，第一个开始标准化的维度
- **epsilon**：数据类型为 float，默认值为 1e-05，指定一个小值与 var 相加，以避免除以 0
- **stash_type**：数据类型为 int，默认值为 1，指定 mean 和 InvStdDev 的类型

## 输出

- **y**：标准化之后的 Tensor，数据类型为 float16 或 float
- **mean**：可选输出，均值
- **InvStdDev**：可选输出，标准差的倒数

## 约束与限制

无

## 支持的 ONNX 版本

Opset v17/v18
