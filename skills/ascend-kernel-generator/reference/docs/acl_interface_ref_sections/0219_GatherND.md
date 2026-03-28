### GatherND

## 功能

对于秩 r ≥ 1 的输入 Tensor `data` 和秩 q ≥ 1 的输入 Tensor `indices`，将数据切片收集到秩为 `(q-1) + (r - indices_shape[-1])` 的输出 Tensor `output` 中。

`indices` 是一个 q 维的整型 Tensor，可视作一个 q-1 维的由索引对构成的特殊张量（每个索引对是一个长度为 `indices_shape[-1]` 的一维张量，每个索引对指向 `self` 中一个切片）。

## 输入

- **data**：输入 Tensor，秩 r ≥ 1，数据类型：int8、uint8、float16、float、double、int32、int64。
- **indices**：输入 Tensor，秩 q ≥ 1，数据类型：int64。

## 属性

- **batch_dims**：数据类型支持 int，批处理轴的数量，当前仅支持配置为 0。

## 输出

- **output**：输出 Tensor，秩为 `(q-1) + (r - indices_shape[-1])`，即 `output` 的 shape 为 `[indices_shape[0:q-1], self_shape[indices_shape[-1]:r]]`。

## 约束

不支持 atc 工具参数 `--precision_mode=must_keep_origin_dtype` 时 double 的输入。

## 支持的 ONNX 版本

Opset v11/v12/v13/v14/v15/v16/v17/v18
