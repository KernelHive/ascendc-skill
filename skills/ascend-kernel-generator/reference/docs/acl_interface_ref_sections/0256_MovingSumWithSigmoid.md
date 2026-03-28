### MovingSumWithSigmoid

## 功能

对输入 `alpha` 进行滑窗求和，输入 `energy` 做 Sigmoid 计算，两者乘积生成输出。

## 输入

- **alpha**：输入 Tensor，shape=[bst]，数据类型支持 float16、float
- **energy**：输入 Tensor，shape=[bst]，数据类型支持 float16、float
- **offset**：输入 Tensor，shape=[2 * batch_size]，数据类型：int32
- **dec_data**：输入 Tensor，shape=[bs, feature_dim]，数据类型支持 float16、float

## 属性

- **ksize**：数据类型为 int

## 输出

- **y**：输出 Tensor，shape=[bs, bt]，数据类型支持 float16、float

## 约束与限制

该算子为特定用户场景定制，非相关场景不建议使用。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16
