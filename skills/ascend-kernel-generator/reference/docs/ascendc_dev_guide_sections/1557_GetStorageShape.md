##### GetStorageShape

## 函数功能
获取运行时shape。

## 函数原型
```cpp
const Shape &GetStorageShape() const
```

## 参数说明
无。

## 返回值说明
运行时shape。

## 约束说明
无。

## 调用示例
```cpp
StorageShape shape({3, 256, 256}, {256, 256, 3});
auto storage_shape = shape.GetStorageShape(); // 256,256,3
```
