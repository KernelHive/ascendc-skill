### ScatterND

## 功能
创建 data 的拷贝，同时在指定 indices 处根据 updates 更新。

## 输入
- **data**：输入 Tensor，数据类型支持 float16、float。
- **indices**：输入 Tensor，`indices.shape[-1] <= data` 维度。数据类型支持 int64。
- **updates**：输入 Tensor，shape 维度 = `data 维度 + indices 维度 - indices_shape[-1] - 1`，数据类型支持 float16、float。

## 输出
- **y**：输出 Tensor，和输入 data 同样的数据类型和 shape。

## 约束
indices 的索引值不能为负数。

## 支持的 ONNX 版本
Opset v11/v12/v13/v14/v15/v16/v17/v18
