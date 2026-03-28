##### SetData

## 函数功能
向 Tensor 中设置数据。

## 函数原型

```cpp
graphStatus SetData(std::vector<uint8_t> &&data)
graphStatus SetData(const std::vector<uint8_t> &data)
graphStatus SetData(const uint8_t *data, size_t size)
graphStatus SetData(const std::string &data)
graphStatus SetData(const char_t *data)
graphStatus SetData(const std::vector<std::string> &data)
graphStatus SetData(const std::vector<AscendString> &datas)
graphStatus SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func)
```

## 须知
数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名         | 输入/输出 | 描述                               |
|----------------|-----------|------------------------------------|
| data/datas     | 输入      | 需设置的数据。                     |
| size           | 输入      | 数据的长度，单位为字节。           |
| deleter_func   | 输入      | 用于释放 data 数据。               |

```cpp
using DeleteFunc = std::function<void(uint8_t *)>;
```

## 返回值
graphStatus 类型：设置成功返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理
无。

## 约束说明
无。
