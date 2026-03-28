## SuperKernel 开发

SuperKernel 是一种算子的二进制融合技术，与源码融合不同，它聚焦于内核函数（Kernel）的二进制的调度方案，展开深度优化，于已编译的二进制代码基础上融合创建一个超级 Kernel 函数（SuperKernel），以调用子函数的方式调用多个其他内核函数，也就是子 Kernel。相对于单算子下发，SuperKernel 技术可以减少任务调度等待时间和调度开销，同时利用 Task 间隙资源进一步优化算子头开销。

## 说明

- SuperKernel 仅适用于静态图场景。
- SuperKernel 仅适用于 Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 自定义算子支持 SuperKernel

自定义算子支持 SuperKernel 与普通算子在开发流程上并无显著差异，但需注意一些特定约束（详见下文）。当前 SuperKernel 特性仅支持在 PyTorch 框架使用，所以完成算子入图（GE 图）开发后，还需要参考《PyTorch 图模式使用指南（TorchAir）》中的“自定义算子入图”章节，完成 PyTorch 入图。同时，TorchAir 提供标定 SuperKernel 范围的能力，用户可根据实际业务需求对融合范围内的算子进行标记和优化配置。具体内容请参考《PyTorch 图模式使用指南（TorchAir）》中的“max-autotune 模式功能 > 图内标定 SuperKernel 范围”章节。

## 开发时的特定约束说明

### 全核同步注意事项

- 自定义算子若进行全核同步，需注意子 Kernel（即该算子）启动的核数与 SuperKernel 的核数一致。若子 Kernel 启动的核数少于 SuperKernel 的核数，全核同步会等待所有核完成，导致卡住超时。

> **注**：SuperKernel 启动核数为子 Kernel 的最大启动核数。假设 SuperKernel 包括算子 a（启动核数为 4）和算子 b（启动核数为 2），此时 SuperKernel 启动核数为 4。

- 使用 `SyncAll` 时，为了解决该问题，可以通过在标定 SuperKernel 范围时开启 `feed-sync-all` 功能，此时系统会在 SuperKernel 内子 Kernel 的其余核中插入 `SyncAll` 指令，防止卡住超时。

- 若使用 `CrossCoreSetFlag` 和 `CrossCoreWaitFlag` 硬同步接口实现全核同步，仅支持子 Kernel 启动核数与 SuperKernel 核数相同。

### 混合 Kernel 类型与硬同步

- 若自定义算子的 Kernel 类型设置为 `KERNEL_TYPE_MIX_AIC_1_1`，并且算子内部使用了 AIC 与 AIV 之间的硬同步接口（`CrossCoreSetFlag` 和 `CrossCoreWaitFlag`），因为 SuperKernel 会根据启动核数等信息调整 SuperKernel 的启动比例，此时需特别注意该算子也可以适应 SuperKernel 的 1:2 启动比例，确保 AIC 与 AIV 之间的硬同步操作正确执行。例如：不单独指定某些 AIV 核调用硬同步接口，使所有 AIV 核均调用硬同步接口，防止因为硬同步数量不匹配而导致卡死超时。

### 数据缓存一致性

- 在开发自定义算子时，开发者必须确保所有对 GM 的标量读写操作都按需正确插入 `DataCacheCleanAndInvalid` 指令：在单算子编译场景下，毕昇编译器自动在算子末尾添加 `DataCacheCleanAndInvalid` 指令，刷新整个 DCache（数据缓存）。在 SuperKernel 中，子 Kernel 被当做普通函数处理，编译器不会自动插入该指令，来确保数据缓存一致性。开发者需要自行保证避免因容错机制改变而导致错误。

### 其他约束

- 不支持使能 Tiling 下沉的自定义算子融合成 SuperKernel。
- 在子 Kernel 中调用 `GetBlockNum` 接口获取核数时，无论是否融合 SuperKernel，获取的核数保持不变，不受 SuperKernel 启动核数的影响。因此，在使用该接口时，开发者无需特别关注 SuperKernel 的启动核数，使用方法和开发普通算子时一样。

## 任务间接口优化性能

此外，开发者在进行 Kernel 侧编程时，可以通过调用 `SetNextTaskStart` 和 `WaitPreTaskEnd` 两个任务间接口，进一步提升性能。

- 调用 `SetNextTaskStart` 后的指令可以和后续其他的子 Kernel 实现并行，提升整体性能。如图 9-2 所示，SuperKernel 按序调用子 Kernel，为保证子 Kernel 之间数据互不干扰，会在子 Kernel 间插入算子间同步进行保序，子 Kernel N-1 调用该接口后，之后的指令会和后续子 Kernel N 实现并行。

> **图 9-2** 通过 SetNextTaskStart 实现并行示意图

- 调用 `WaitPreTaskEnd` 前的指令可以和前序其他的子 Kernel 实现并行，提升整体性能。如图 9-3 所示，SuperKernel 按序调用子 Kernel，为保证子 Kernel 之间数据互不干扰，会在子 Kernel 间插入算子间同步进行保序，子 Kernel N+1 调用该接口之前的指令会和前序子 Kernel N 实现并行。

> **图 9-3** 通过 WaitPreTaskEnd 实现并行示意图
