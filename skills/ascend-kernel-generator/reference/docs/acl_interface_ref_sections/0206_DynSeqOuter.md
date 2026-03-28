### DynSeqOuter

## 功能
按照偏移量求和。

## 输入
- **x1**：输入Tensor
  - Shape: `[bs, feature_dim]`
  - bs：单批数据个数
  - feature_dim：特征维度
  - 数据类型：float16、float
- **x2**：输入Tensor
  - Shape: `[bs, feature_dim]`
  - bs：单批数据个数
  - feature_dim：特征维度
  - 数据类型：float16、float
- **seq_len1**：输入Tensor
  - Shape: `[batch_size]` 且 `bs = sum(seq_len1)`
  - 数据类型：int32
- **seq_len2**：输入Tensor
  - Shape: `[batch_size]` 且 `bs = sum(seq_len2)`
  - 数据类型：int32

## 输出
- **y**：输出Tensor
  - Shape: `[bst, feature_dim]` 且 `bs = sum(seq_len1 * seq_len2)`
  - 数据类型：float16、float

## 约束
该算子为特定用户场景定制，非相关场景不建议使用。

## 支持的 ONNX 版本
Opset v8/v9/v10/v11/v12/v13/v14/v15/v16
