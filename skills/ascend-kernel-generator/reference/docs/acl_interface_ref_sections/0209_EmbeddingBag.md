### EmbeddingBag

## 功能

根据 `indices` 从 `weight` 中获得一组被聚合的数，然后根据 `offsets` 的偏移和 `mode` 指定的聚合模式对获取的数进行 max、sum、mean 聚合。其余参数则更细化了计算过程的控制。

## 输入

- **weight**：输入 Tensor，词嵌入矩阵，包含所有词的嵌入向量。
  - 支持 2 维
  - 支持非连续 Tensor
  - 数据类型：float
  - shape：支持 ND 数据格式

- **indices**：输入 Tensor，包含索引的 Tensor，指定要从 `weight` 中提取哪些词的嵌入向量。
  - 支持 1-2 维
  - 数据类型：UINT8、INT8、INT16、INT32、INT64
  - shape：支持 ND 数据格式

- **offset**（可选）：输入 Tensor，用于将 `indices` 分割成多个 bag 的偏移量张量。
  - 当 `indices` 是 1 维时，`offsets` 的 shape 支持 1 维
  - 当 `indices` 是 2 维时，`offsets` 的 shape 支持 1-2 维
  - 数据类型：UINT8、INT8、INT16、INT32、INT64

- **per_sample_weights**（可选）：指定样本权重。
  - shape 支持 1 维
  - 数据类型与 `weight` 一致
  - 仅在 sum 模式下可以不是 nullptr，其他模式必须为 nullptr

## 属性

- **mode**：数据类型支持 string 型，用于控制聚合模式。
  - 0 表示 sum 聚合模式
  - 1 表示 mean 聚合模式
  - 其他表示 max 聚合模式

- **scale_grad_by_freq**：数据类型支持 bool 型，输入 Tensor，用于控制是否根据词频缩放梯度。
  - 当 `scale_grad_by_freq` 为 true 时，会根据词频对梯度进行缩放
  - 当 `scale_grad_by_freq` 为 false 时，则不会

- **sparse**：数据类型支持 bool 型，用于控制稀疏模式。
  - 当为 false 时，表示 weight 非稀疏矩阵
  - 当为 true 时，表示 weight 是稀疏矩阵

- **include_last_offset**：数据类型支持 bool 型，控制是否包含最后的偏移。
  - 当为 false 时，表示不包含最后的偏移
  - 当为 true 时，表示包含最后的偏移

## 输出

- **y**：输出 Tensor
  - 数据类型：float

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16
