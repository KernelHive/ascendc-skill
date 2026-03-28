### RandomNormal

## 功能

生成具有从正态分布绘制的随机值的 Tensor。

## 属性

- **dtype**：数据类型为 int，默认值为 `TensorProto::FLOAT`，输出 Tensor 元素的数据类型。
- **mean**：数据类型为 float，默认值为 `0.0`，正态分布的平均值。
- **scale**：数据类型为 float，默认值为 `1.0`，正态分布的标准偏差。
- **seed**（可选）：数据类型为 float，默认为 `0`，指定一个随机数种子。
- **shape**：数据类型为 int 列表，输出 Tensor 的形状。

## 输出

**y**：输出 Tensor，数据类型支持 `float16`、`float32`、`double`。

## 约束与限制

无。

## 支持的 ONNX 版本

v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18。
