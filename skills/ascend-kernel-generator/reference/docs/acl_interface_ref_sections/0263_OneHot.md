### OneHot

## 功能

对长度为 n 的输入 Tensor，经过 one_hot 的计算后得到一个元素数量为 n × k 的输出 Tensor，其中 k 的值为 numClasses。

## 输入

- **indices**：输入 Tensor，数据类型支持 int32
- **depth**：输入 Tensor，数据类型支持 int32
- **values**：输入 Tensor，数据类型支持 int32、float16、float

## 属性

- **axis**（可选）：数据类型为 int64，添加算子表示的轴

## 输出

- **y**：输出 Tensor，数据类型与输入 indices 一致

## 约束与限制

- 算子属性不支持 axis < -1
- depth ≥ 1
- 当输出为 int64 时，最后一个输入值 values = [0, 1]，1 代表 on_value，0 代表 off_value

## 支持的 ONNX 版本

Opset v9/v10/v11/v12/v13/v14/v15/v16/v17/v18
