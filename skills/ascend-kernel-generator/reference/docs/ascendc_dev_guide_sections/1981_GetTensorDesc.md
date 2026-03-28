##### GetTensorDesc

## 函数功能
获取 Tensor 的描述符。

## 函数原型
```cpp
TensorDesc GetTensorDesc() const
```

## 参数说明
无。

## 返回值
返回当前 Tensor 的描述符，类型为 `TensorDesc`。

## 异常处理
无。

## 约束说明
修改返回的 `TensorDesc` 信息，不会影响 Tensor 对象中已有的 `TensorDesc` 信息。
