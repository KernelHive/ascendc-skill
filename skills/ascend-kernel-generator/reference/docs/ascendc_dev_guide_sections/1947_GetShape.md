##### GetShape

## 函数功能
获取 TensorDesc 所描述 Tensor 的 Shape。

## 函数原型
```cpp
Shape GetShape() const
```

## 参数说明
无。

## 返回值
Shape 类型，TensorDesc 描述的 Shape。

## 异常处理
无。

## 约束说明
由于返回的 Shape 信息为值拷贝，因此修改返回的 Shape 信息，不影响 TensorDesc 中已有的 Shape 信息。
