##### MutableStorageShape

## 函数功能
获取可写的运行时shape。

## 函数原型
```cpp
Shape &MutableStorageShape()
```

## 参数说明
无。

## 返回值说明
可写的运行时shape。

## 约束说明
无。

## 调用示例
```cpp
StorageShape shape({3, 256, 256}, {256, 256, 3});
auto storage_shape = shape.MutableStorageShape(); // 256,256,3
```
