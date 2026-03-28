### SoftmaxCrossEntropyLoss

## 功能
损失函数，用于计算 score 和 label 之间的 SoftmaxCrossEntropy 损失。

## 输入

### scores
输入 Tensor，数据类型支持 float16、float、double，shape 为 `[N, C]` 或 `[N, C, D1, D2, …, Dk]`，其中 N 为 batch，C 为 class。

### labels
输入 Tensor，数据类型支持 int32、int64，shape 与 scores 保持一致。

### weights
输入 Tensor，数据类型支持 float16、float，用于调整不同类别的损失权重，shape 为 `[C]`。

## 属性

### ignore_index
int 类型，用于指定一个目标值，该值将被忽略，不会对梯度计算产生影响。

### reduction
string 类型，用于指定损失的 reduction 类型，可选值包括：
- `"none"`
- `"sum"`
- `"mean"`

## 输出

### output
输出 Tensor，数据类型与 scores 保持一致。若 reduction 为 `"none"`，输出 shape 与 scores 和 labels 保持一致；否则，为 Scalar。

### log_prob
输出 Tensor，数据类型与 scores 保持一致。表示 Softmax 输出的概率值的对数。

## 支持的 ONNX 版本
Opset v12/v13/v14/v15/v16/v17/v18
