##### InferenceContext 构造函数和析构函数

## 函数功能

InferenceContext 对象的构造函数和析构函数。

## 函数原型

```cpp
~InferenceContext() = default
InferenceContext(const InferenceContext &context) = delete
InferenceContext(const InferenceContext &&context) = delete
InferenceContext &operator=(const InferenceContext &context) = delete
InferenceContext &operator=(const InferenceContext &&context) = delete
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| context | 输入 | InferenceContext 内容，供初始化使用。 |

## 返回值

InferenceContext 构造函数返回 InferenceContext 类型的对象。

## 异常处理

无。

## 约束说明

无。
