##### GetInferenceContext

## 功能
获取 InferenceContext 指针。

## 函数原型
```cpp
ge::InferenceContext *GetInferenceContext() const
```

## 参数
无。

## 返回值
输出 InferenceContext 指针。

关于 InferenceContext 类型的定义，请参见 [4.7-InferenceContext](#4.7-InferenceContext)。

## 约束
无。

## 调用示例
```cpp
ge::graphStatus InferShapeForXXX(CtInferShapeRangeContext *context) {
    const auto &read_inference_context = ct_context->GetInferenceContext();
    // ...
}
```
