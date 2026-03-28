##### SetInferenceContext

## 函数功能

向当前算子传递 infershape 推导所需要的关联信息，比如前面算子的 shape 和 DataType 信息。

## 函数原型

```cpp
void SetInferenceContext(const InferenceContextPtr &inference_context)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| `inference_context` | 输入 | 当前 operator 的推理上下文。<br>InferenceContextPtr 是指向 InferenceContext 类的指针的别名：<br>`using InferenceContextPtr = std::shared_ptr<InferenceContext>;` |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
