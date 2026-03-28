##### SaveToFile

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

将序列化后的Graph结构保存到文件中。

## 函数原型

```cpp
graphStatus SaveToFile(const std::string &file_name) const
graphStatus SaveToFile(const char_t *file_name) const
```

### 须知

数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| file_name | 输入 | 需要保存的文件路径和文件名。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功。<br>其他值：失败。 |

## 约束说明

保存的文件格式为air格式，可以通过15.2.3.9.16 LoadFromFile接口还原为graph对象。
