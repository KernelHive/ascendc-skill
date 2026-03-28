##### InputTensorsRange

## 函数功能

设置输入 Tensor 的 Range 指针，用于在 Shape Range 推导时，可通过该 Builder 类构造的上下文 `InferShapeRangeContext` 获取相应的输入 Tensor Range 指针，即可以获得最大 Shape 的 Tensor 和最小 Shape 的 Tensor。

## 函数原型

```cpp
OpInferShapeRangeContextBuilder &InputTensorsRange(
    const std::vector<gert::Range<gert::Tensor> *> &inputs
)
```

## 参数说明

| 参数   | 输入/输出 | 说明                                                                 |
| ------ | --------- | -------------------------------------------------------------------- |
| inputs | 输入      | `gert::Range<gert::Tensor> *` 类型的数组，存储各算子输入的 Tensor Range 指针，Tensor Range 包含最大 Shape 的 Tensor 和最小 Shape 的 Tensor。 |

## 返回值说明

`OpInferShapeRangeContextBuilder` 对象本身，用于链式调用。

## 约束说明

- 在调用 `Build` 方法之前，必须调用该接口，否则构造出的 `InferShapeRangeContext` 将包含未定义数据。
- 通过指针传入的参数（`gert::Tensor *`），其内存所有权归调用者所有；调用者必须确保指针在 `ContextHolder` 对象的生命周期内有效。
