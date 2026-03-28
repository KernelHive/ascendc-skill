##### CreateCustomValue

## 函数功能

创建一个类型为 `kCustomValue`（自定义参数类型）的 `ArgDescInfo` 对象，表示此块 Args 地址用来存储自定义内容。

## 函数原型

```cpp
static ArgDescInfo CreateCustomValue(uint64_t custom_value)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| custom_value | 输入 | Args 中此块内存中保存的自定义数据 |

## 返回值说明

返回一个 `ArgDescInfo` 对象，此对象的 `type` 为 `kCustomValue`。

## 约束说明

无。

## 调用示例

```cpp
// 需要存储在 Args 中的结构体
struct HcclCommParamDesc {
    uint64_t version : 4;
    uint64_t group_num : 4;
    uint64_t has_ffts : 1;
    uint64_t tiling_off : 7;
    uint64_t is_dyn : 48;
};

graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
    ...
    // 设置 AI CPU 任务
    auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context,
        "libccl_kernel.so", "RunAicpuKfcSrvLaunch");
    size_t input_size = context->GetComputeNodeInfo()->GetIrInputsNum();
    size_t output_size = context->GetComputeNodeInfo()->GetIrOutputsNum();

    const size_t offset = 3UL;
    union {
        HcclCommParamDesc hccl_desc;
        uint64_t custom_value;
    } desc;
    
    // 赋值
    desc.hccl_desc.version = 1;
    desc.hccl_desc.group_num = 1;
    desc.hccl_desc.has_ffts = 0;
    desc.hccl_desc.tiling_off = offset + input_size + output_size;
    desc.hccl_desc.is_dyn = 0;
    
    std::vector<ArgDescInfo> aicpu_args_format;
    // 将此结构体的内容转化成 uint64_t 的数字保存到 ArgsFormat 中
    aicpu_args_format.emplace_back(ArgDescInfo::CreateCustomValue(desc.custom_value));
    ...
}
```
