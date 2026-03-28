##### CreateHiddenInput

## 功能

创建一个类型为 `kHiddenInput`（在 IR 原型定义上不存在的输入地址）的 `ArgDescInfo` 对象，表示此块 Args 地址为 HiddenInput 地址。

## 函数原型

```cpp
static ArgDescInfo CreateHiddenInput(HiddenInputSubType hidden_type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| `hidden_type` | 输入 | HiddenInput 的类型，`HiddenInputSubType` 类型 |

## 返回值

返回一个 `ArgDescInfo` 对象，此对象的 `type` 为 `kHiddenInput`。

## 约束

无

## 调用示例

```cpp
graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
                               std::vector<std::vector<uint8_t>> &tasks) {
  ...
  // 设置 AI CPU 任务
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context,
                                                         "libccl_kernel.so",
                                                         "RunAicpuKfcSrvLaunch");
  std::vector<ArgDescInfo> aicpu_args_format;
  // 创建一个类型是 Hcom 的 HiddenInput 地址
  aicpu_args_format.emplace_back(ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
  ...
}
```
