### Multinomial

## 功能

在输入 `x` 中根据每个对象分布的概率，抽取 `numSamples` 个样本，并将这些样本的索引存储在输出 `y` 中。

## 输入

- **x**：输入 Tensor，shape = `[batch_size, class_size]`，数据类型支持 float16、float。
  - `class_size` 指所有可能结果的数量，每个值表示该 batch 中每个相应结果的非归一化对数概率。

## 属性

- **dtype**：数据类型为 int，默认为 6，输出数据类型。
- **sample_size**：数据类型为 int，默认为 1，采样次数。
- **seed**：数据类型为 float，随机数种子。

## 输出

- **y**：输出 Tensor，shape = `[batch_size, sample_size]`，数据类型支持 int32、int64。
  - `sample_size` 指采样的次数，每个值表示该 batch 中相应样本的结果。

## 说明

如果调用该算子超时，需要使用 `ret = acl.rt.set_op_execute_time_out(timeout)` 接口避免超时。

## 约束与限制

无。

## 支持的 ONNX 版本

Opset v8/v9/v10/v11/v12/v13/v14/v15/v16/v17/v18。
