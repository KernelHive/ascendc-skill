##### Tensor 构造函数和析构函数

## 函数功能
Tensor 构造函数和析构函数。

## 函数原型
```cpp
Tensor()
~Tensor() = default
explicit Tensor(const TensorDesc &tensor_desc)
Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data)
Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size)
Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data)
```

## 参数说明

| 参数名       | 输入/输出 | 描述                                      |
|--------------|-----------|-------------------------------------------|
| tensor_desc  | 输入      | TensorDesc 对象，需设置的 Tensor 描述符。 |
| data         | 输入      | 需设置的数据。                            |
| size         | 输入      | 数据的长度，单位为字节。                  |

## 返回值
Tensor 构造函数返回 Tensor 类型的对象。

## 异常处理
无。

## 约束说明
无。
