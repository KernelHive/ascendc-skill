##### GetCustomValue

## 功能描述
获取 ArgDescInfo 的自定义值。

> **注意**：只有当 `type` 为 `kCustomValue` 时，才能获取到正确的值。

## 函数原型
```cpp
uint64_t GetCustomValue() const
```

## 参数
无

## 返回值
返回自定义内容的值，默认值为 `0`。

> **异常情况**：获取到的值为 `uint64_max`。

## 约束
无

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
    // 此处 custom_value 中的值便是 HcclCommParamDesc 内容拼接而成的结果
    uint64_t custom_value = aicpu_args_format.back().GetCustomValue();
    ...
}
```
