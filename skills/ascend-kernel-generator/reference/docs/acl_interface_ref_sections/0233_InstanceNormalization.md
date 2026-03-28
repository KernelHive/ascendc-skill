### InstanceNormalization

## 功能
实例归一化运算符。

其中 mean 和 variance 是每个实例每个通道的均值和方差。

## 输入
- **x**：输入 Tensor，数据类型支持 float16、float。
- **scale**：1 维输入 Tensor，维度同 x 的 C 轴长度，和输入 x 同样的数据类型。
- **B**：1 维输入 Tensor，维度同 x 的 C 轴长度，和输入 x 同样的数据类型。

## 属性
- **epsilon**：数据类型为 float，默认是 1e-05，避免除 0。

## 输出
- **y**：输出 Tensor，数据类型和 shape 与输入一致。

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
