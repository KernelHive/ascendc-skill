##### PlatformInfo

## 函数功能
设置 PlatformInfo 平台信息。

## 函数原型
```cpp
OpTilingContextBuilder &PlatformInfo(const void *platform_info)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| platform_info | 输入 | 平台信息指针。 |

## 返回值说明
返回 OpTilingContextBuilder 对象本身，用于链式调用。

## 约束说明
- 在调用 `Build` 方法之前，必须调用本接口，否则构造出的 TilingContext 将包含未定义数据。
- 通过指针传入的参数（`void*`），其内存所有权归调用者所有；调用者必须确保指针在 ContextHolder 对象的生命周期内有效。
