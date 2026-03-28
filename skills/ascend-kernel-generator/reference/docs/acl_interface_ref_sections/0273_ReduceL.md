### ReduceL

## 功能
根据所提供的轴计算输入 Tensor 元素的 L1 范数（即元素绝对值的和）。

如果 `keepdims` 等于 1，得到的 Tensor 的秩与输入的相同。如果 `keepdims` 等于 0，那么得到的 Tensor 就会被精简维数。上述行为与 numpy 类似，只是 numpy 默认 `keepdims` 为 `False` 而不是 `True`。

## 输入
- **x**：输入 Tensor，数据类型支持 float16、float、double。

## 属性
- **axes**：数据类型为 int 列表，指定计算轴。
- **keepdims**：数据类型为 int，默认值为 1，是否保留缩减后的维度。

## 输出
- **y**：输出 Tensor，数据类型支持 float16、float、double。

## 约束与限制
无。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16。
