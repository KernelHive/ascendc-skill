##### GetInferenceContext

## 函数功能

获取当前算子传递 infershape 推导所需要的关联信息，比如前面算子的 shape 和 DataType 信息。

## 函数原型

```cpp
InferenceContextPtr GetInferenceContext() const
```

## 参数说明

无。

## 返回值

返回当前 operator 的推理上下文。

`InferenceContextPtr` 是指向 `InferenceContext` 类的指针的别名：

```cpp
using InferenceContextPtr = std::shared_ptr<InferenceContext>;
```

## 异常处理

无。

## 约束说明

无。
