###### PlatformInfo

## 功能说明

将指向 `fe::PlatFormInfos` 的指针传入 `TilingContext`。

## 函数原型

```cpp
ContextBuilder &PlatformInfo(void *platformInfo)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| platformInfo | 输入 | 指向 `fe::PlatFormInfos` 类型数据的 void 指针 |

## 返回值说明

返回当前 `ContextBuilder` 的对象。

## 约束说明

由于 `TilingContext` 与 `KernelContext`、`TilingParseContext` 内部数据排序不同，`Platform()` 只支持以调用 `BuildTilingContext()` 为前提来使用；其他场景建议用 `Outputs` 接口，否则发生未定义行为。

## 调用示例

```cpp
void AddPlatformInfo(fe::PlatFormInfos *platformInfo)
{
    // ...
    auto kernelContextHolder = context_ascendc::ContextBuilder()
        // ... 增加算子输入输出接口的调用
        .PlatformInfo(reinterpret_cast<void*>(platformInfo))
        .BuildTilingContext();
    // ...
}
```
