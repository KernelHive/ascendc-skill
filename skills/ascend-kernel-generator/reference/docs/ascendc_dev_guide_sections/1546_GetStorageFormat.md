##### GetStorageFormat

## 函数功能
获取运行时 format。

## 函数原型
```cpp
ge::Format GetStorageFormat() const
```

## 参数说明
无。

## 返回值说明
运行时 format。

## 约束说明
无。

## 调用示例
```cpp
ExpandDimsType dim_type("1100");
StorageFormat format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_C1HWNC0, dim_type);
auto storage_format = format.GetStorageFormat(); // Format::FORMAT_C1HWNC0
```
