##### SetConstData

## 函数功能
如果 TensorDesc 是常量节点的描述，向 TensorDesc 中设置权重值。

## 函数原型
```cpp
void SetConstData(std::unique_ptr<uint8_t[]> const_data_buffer, const size_t &const_data_len)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| `const_data_buffer` | 输入 | 权重地址。 |
| `const_data_len` | 输入 | 权重长度。 |

## 返回值
无。

## 异常处理
无。

## 约束说明
无。
