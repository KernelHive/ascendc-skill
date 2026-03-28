##### SetMarks

## 函数功能

在资源类算子推理的上下文中，设置成对资源算子的标记。

## 函数原型

```cpp
void SetMarks(const std::vector<std::string> &marks)
void SetMarks(const std::vector<AscendString> &marks)
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| marks  | 输入      | 资源类算子的标记。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
