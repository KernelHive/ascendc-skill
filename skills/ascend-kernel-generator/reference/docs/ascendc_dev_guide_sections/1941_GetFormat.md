##### GetFormat

## 函数功能
获取 TensorDesc 所描述的 Tensor 的 Format。

## 函数原型
```cpp
Format GetFormat() const
```

## 参数说明
无。

## 返回值
Format 类型，TensorDesc 所描述的 Tensor 的 Format 信息。

## 异常处理
无。

## 约束说明
由于返回的 Format 信息为值拷贝，因此修改返回的 Format 信息，不影响 TensorDesc 中已有的 Format 信息。
