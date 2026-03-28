##### IsScalar

## 功能

判断当前形状是否为标量。标量是指 `GetDimNum()` 返回值为 0 的形状。

## 函数原型

```cpp
bool IsScalar() const
```

## 参数

无。

## 返回值

- `true`：形状为标量
- `false`：形状不为标量

## 约束

无。

## 示例

```cpp
Shape shape0({3, 256, 256});
Shape shape2;
shape0.IsScalar(); // 返回 false
shape2.IsScalar(); // 返回 true
```
