##### IsValid

## 函数功能
判断 Tensor 对象是否有效。  
若实际 Tensor 数据的大小与 TensorDesc 所描述的 Tensor 数据大小一致，则有效。

## 函数原型
```cpp
graphStatus IsValid()
```

## 参数说明
无。

## 返回值
`graphStatus` 类型：  
如果 Tensor 对象有效，则返回 `GRAPH_SUCCESS`，否则返回 `GRAPH_FAILED`。

## 异常处理
无。

## 约束说明
无。
