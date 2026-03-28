##### PlatformInfo

## 函数功能

设置算子的 PlatformInfo 指针，用于构造 TilingParseContext 的 PlatformInfo 字段。

## 函数原型

```cpp
OpTilingParseContextBuilder &PlatformInfo(const void *platform_info)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| platform_info | 输入 | 平台信息指针 |

## 返回值说明

返回 OpTilingParseContextBuilder 对象本身，用于链式调用。

## 约束说明

通过指针传入的参数（void*），其内存所有权归调用者所有；调用者必须确保指针在 ContextHolder 对象的生命周期内有效。
