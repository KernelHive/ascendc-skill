##### SetDimNum

## 函数功能

设置 Shape 的维度个数。

## 函数原型

```cpp
void SetDimNum(const size_t dim_num)
```

## 参数说明

| 参数     | 输入/输出 | 说明                         |
|----------|-----------|------------------------------|
| dim_num  | 输入      | 将 Shape 的维度个数设置为 dim_num。 |

## 返回值说明

无。

## 约束说明

无。

## 调用示例

```cpp
Shape shape0({3, 256, 256});
shape0.SetDimNum(1);
auto dim_num = shape0.GetDimNum(); // 1
```
