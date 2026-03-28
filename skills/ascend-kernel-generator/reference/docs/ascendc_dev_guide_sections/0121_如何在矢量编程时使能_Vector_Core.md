## 如何在矢量编程时使能 Vector Core

Atlas 推理系列产品在硬件架构中除了 AI Core 外，还设置了独立的 Vector Core，作为 AI Core 中 Vector 计算单元的补充，以缓解 Vector 计算瓶颈。Vector Core 仅包含两种基础计算资源：

- **向量计算单元（Vector Unit）**：用于完成向量数据计算
- **标量计算单元（Scalar Unit）**：用于完成标量数据计算

在矢量算子开发时，使能 Vector Core 后，算子执行会同时启动 AI Core 和 Vector Core，这些核并行执行相同的核函数代码。

> 学习本节内容前，建议先熟悉以下内容：
> - 算子实现
> - 6.6 Kernel 直调算子开发
> - 6.7 工程化算子开发
> - 基于 AI Core 的算子端到端开发流程

本章重点阐述使能 Vector Core 时的差异点，具体如下：

## 1. Kernel 侧开发

完成算子 kernel 侧开发时，需通过宏 `KERNEL_TASK_TYPE_DEFAULT` 使能 Vector Core。此时 AI Core 会作为 Vector Core 使用。

以下代码展示了使能 Vector Core 的方法：

```cpp
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR usr = AscendC::GetUserWorkspace(workspace);
    KernelAdd op;
    op.Init(x, y, z, tilingData.blockDim, tilingData.totalLength, tilingData.tileNum);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE); // 使能 Vector Core
    if (TILING_KEY_IS(1)) {
        op.Process1();
    } else if (TILING_KEY_IS(2)) {
        op.Process2();
    }
    // ...
}
```

## 2. Host 侧 Tiling 开发

完成 host 侧 tiling 开发时，设置的 `blockDim` 代表 AI Core 和 Vector Core 的总数。例如，设置 `blockDim` 为 10，则会启动总数为 10 的 AI Core 和 Vector Core。

为保证启动 Vector Core，设置数值应大于 AI Core 的核数。可通过以下接口获取核数：

- `GetCoreNumAic()`：获取 AI Core 核数
- `GetCoreNumVector()`：获取 Vector Core 核数

以下为两种工程中的设置示例，`blockDim` 设置为 AI Core 和 Vector Core 的总和，表示所有核都启动。

### Kernel 直调工程

```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
auto totalCoreNum = ascendcPlatform.GetCoreNumAic();
// ASCENDXXX 请替换为实际的版本型号
if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCENDXXX) {
    totalCoreNum = totalCoreNum + ascendcPlatform.GetCoreNumVector();
}
...
kernel_name<<<totalCoreNum, l2ctrl, stream>>>(argument list);
```

### 自定义算子工程

```cpp
// 配套的 host 侧 tiling 函数示例：
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 使能 Vector Core，将 blockDim 置为 AI Core 中 vector 核数 + Vector Core 中的 vector 核数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto totalCoreNum = ascendcPlatform.GetCoreNumAic();
    // ASCENDXXX 请替换为实际的版本型号
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCENDXXX) {
        totalCoreNum = totalCoreNum + ascendcPlatform.GetCoreNumVector();
    }
    context->SetBlockDim(totalCoreNum);
}
```

## 说明

- 请参考 Ascend C API 中具体 API 支持的型号，判断接口是否支持 Atlas 推理系列产品 Vector Core。
- 支持 Vector Core 后，AI Core 和 Vector Core 分别执行，通过不同任务调度，因此不支持核间同步指令，如 `IBSet`、`IBWait`、`SyncAll` 等。
- 算子计算溢出（输入 inf/nan 或计算结果超出范围）时，需注意 AI Core 和 Vector Core 结果表现不一致：
  - AI Core 仅支持饱和模式
  - Vector Core 仅支持 inf/nan 模式
