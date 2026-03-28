##### LoadFromFile

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

从文件中读取Graph。

## 函数原型

> **须知**
> 
> 数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```cpp
graphStatus LoadFromFile(const std::string &file_name)
graphStatus LoadFromFile(const char_t *file_name)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| file_name | 输入 | 文件路径和文件名 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功<br>其他值：失败 |

## 约束说明

仅支持读取air格式的文件。
