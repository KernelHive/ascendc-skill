##### AppendDim

## 函数功能

向后扩展一个 dim 值。如果扩展的 dim 数量超出 Shape 的最大限制，那么本函数不执行任何操作。

## 函数原型

```cpp
Shape& AppendDim(const int64_t value)
```

## 参数说明

| 参数  | 输入/输出 | 说明           |
|-------|-----------|----------------|
| value | 输入      | 扩展的 dim 值。 |

## 返回值说明

返回 `this` 引用。

## 约束说明

无。

## 调用示例

```cpp
Shape shape0({3, 256, 256});
shape0.AppendDim(1024); // 3,256,256,1024
```
