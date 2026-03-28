### MeanVarianceNormalization

## 功能
对输入 Tensor x 进行均值方差标准化。

## 输入
- **x**：输入 Tensor 列表，数据类型支持 float16、float。

## 属性
- **axes**：数据类型为 list of ints，默认值：[0, 2, 3]。

## 输出
- **y**：输出 Tensor 列表，数据类型：float16、float。

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
