##### GetFormat

## 函数功能
获取编译时 Tensor 的数据排布格式。

## 函数原型
```cpp
const StorageFormat &GetFormat() const
```

## 参数说明
无。

## 返回值说明
返回数据排布格式。StorageFormat 类型，包括原始 format 和存储 format。

## 约束说明
无。

## 调用示例
```cpp
StorageFormat storageFormat(ge::Format::FORMAT_NC, ge::FORMAT_NCHW, {});
gert::CompileTimeTensorDesc compileTimeTensorDesc;
compileTimeTensorDesc.SetStorageFormat(storageFormat.GetStorageFormat());
auto storage_fmt = compileTimeTensorDesc.GetFormat(); // ge::FORMAT_NCHW
compileTimeTensorDesc.SetOriginFormat(storageFormat.GetOriginFormat());
auto origin_fmt = compileTimeTensorDesc.GetOriginFormat(); // ge::Format::FORMAT_NC
```
