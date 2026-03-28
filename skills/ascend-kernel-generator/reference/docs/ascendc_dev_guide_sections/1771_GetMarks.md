##### GetMarks

## 函数功能

在资源类算子推理的上下文中，获取成对资源算子的标记。

## 函数原型

```cpp
const std::vector<std::string> &GetMarks() const
void GetMarks(std::vector<AscendString> &marks) const
```

## 须知

数据类型为 string 的接口后续版本会废弃，建议使用数据类型为非 string 的接口。

## 参数说明

| 参数 | 输入/输出 | 描述 |
|------|-----------|------|
| marks | 输出 | 资源类算子的标记。 |

## 返回值

资源类算子的标记。

## 异常处理

无。

## 约束说明

无。
