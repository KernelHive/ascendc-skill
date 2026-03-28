##### GetInferenceContext

## 函数功能
获取 InferenceContext 指针。

## 函数原型
```cpp
ge::InferenceContext *GetInferenceContext() const
```

## 参数说明
无。

## 返回值说明
输出 InferenceContext 指针。

关于 InferenceContext 类型的定义，请参见 [4.7-InferenceContext](#4.7-InferenceContext)。

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus InferShapeForXXX(CtInferShapeContext *context) {
    const auto &read_inference_context = ct_context->GetInferenceContext();
    // ...
}
```
