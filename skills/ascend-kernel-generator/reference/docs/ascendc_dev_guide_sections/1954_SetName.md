##### SetName

## 函数功能

向 TensorDesc 中设置 Tensor 的名称。

## 函数原型

```cpp
void SetName(const std::string &name)
void SetName(const char_t *name)
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| name   | 输入      | 需设置的 Tensor 的名称。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
