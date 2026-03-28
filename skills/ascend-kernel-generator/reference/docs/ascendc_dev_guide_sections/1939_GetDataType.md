##### GetDataType

## 函数功能
获取 TensorDesc 所描述 Tensor 的数据类型。

## 函数原型
```cpp
DataType GetDataType() const
```

## 参数说明
无。

## 返回值
DataType 类型，表示 TensorDesc 所描述的 Tensor 的数据类型。

## 异常处理
无。

## 约束说明
由于返回的 DataType 信息为值拷贝，因此修改返回的 DataType 信息，不影响 TensorDesc 中已有的 DataType 信息。
