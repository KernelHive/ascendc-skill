###### ReserveLocalMemory

## 功能说明

该函数用于在 Unified Buffer 中预留指定大小的内存空间。调用该接口后，使用 `GetCoreMemSize` 可以获取实际可用的剩余 Unified Buffer 空间大小。

## 函数原型

```cpp
void ReserveLocalMemory(ReservedSize size)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| `ReservedSize` | 输入 | 需要预留的空间大小。 |

```cpp
enum class ReservedSize {
    RESERVED_SIZE_8K,  // 预留 8 * 1024B 空间
    RESERVED_SIZE_16K, // 预留 16 * 1024B 空间
    RESERVED_SIZE_32K  // 预留 32 * 1024B 空间
};
```

## 返回值说明

无

## 约束说明

多次调用该函数时，仅保留最后一次调用的结果。

## 调用示例

```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size, l1_size;
    // 预留 8KB 的 Unified Buffer 内存空间
    ascendcPlatform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_8K);
    // 获取 Unified Buffer 和 L1 的实际可用内存大小
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
    // ...
    return ret;
}
```

完整样例可参考与数学库高阶 API 配合使用的样例。
