##### SetOutputHandleShapesAndTypes

## 函数功能

在推理上下文中，设置算子输出句柄的 ShapeAndType。

## 函数原型

```cpp
void SetOutputHandleShapesAndTypes(const std::vector<std::vector<ShapeAndType>> &shapes_and_types)
void SetOutputHandleShapesAndTypes(std::vector<std::vector<ShapeAndType>> &&shapes_and_types)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| shapes_and_types | 输入 | 算子输出句柄的 ShapeAndType |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
