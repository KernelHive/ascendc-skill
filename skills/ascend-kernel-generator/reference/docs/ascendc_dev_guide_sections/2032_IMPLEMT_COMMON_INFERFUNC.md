#### IMPLEMT_COMMON_INFERFUNC

## 函数功能

封装算子的 Common_InferShape 函数。

与 `IMPLEMT_INFERFUNC` 的区别是，此函数自动生成一个类型为 `Operator` 类的对象 `op`，可直接调用 `Operator` 接口进行 InferShape 的实现。

若 InferShape 方法具有通用性，可被多个算子的原型实现调用，可选择此接口实现。

## 函数原型

```cpp
IMPLEMT_COMMON_INFERFUNC(func_name)
```

## 约束说明

无。

## 参数说明

| 参数名    | 输入/输出 | 描述                         |
|-----------|-----------|------------------------------|
| func_name | 输入      | InferShape 函数名，用户自定义 |

## 返回值

无。
