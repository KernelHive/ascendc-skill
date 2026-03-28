##### SetBlockDim

## 函数功能
设置 blockDim，即参与计算的 Vector 或者 Cube 核数。

## 函数原型
```cpp
ge::graphStatus SetBlockDim(const uint32_t block_dim)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| block_dim | 输入 | blockDim 是逻辑核的概念，取值范围为 [1,65535]。为了充分利用硬件资源，一般设置为物理核的核数或其倍数。 |

- 对于耦合模式和分离模式，blockDim 在运行时的意义和设置规则有一些区别，具体说明如下：
  - **耦合模式**：由于其 Vector、Cube 单元是集成在一起的，blockDim 用于设置启动多个 AI Core 核实例执行，不区分 Vector、Cube。AI Core 的核数可以通过 `GetCoreNumAiv` 或者 `GetCoreNumAic` 获取。
  - **分离模式**：
    - 针对仅包含 Vector 计算的算子，blockDim 用于设置启动多少个 Vector（AIV）实例执行，比如某款 AI 处理器上有 40 个 Vector 核，建议设置为 40。
    - 针对仅包含 Cube 计算的算子，blockDim 用于设置启动多少个 Cube（AIC）实例执行，比如某款 AI 处理器上有 20 个 Cube 核，建议设置为 20。
    - 针对 Vector/Cube 融合计算的算子，启动时，按照 AIV 和 AIC 组合启动，blockDim 用于设置启动多少个组合执行，比如某款 AI 处理器上有 40 个 Vector 核和 20 个 Cube 核，一个组合是 2 个 Vector 核和 1 个 Cube 核，建议设置为 20，此时会启动 20 个组合，即 40 个 Vector 核和 20 个 Cube 核。注意：该场景下，设置的 blockDim 逻辑核的核数不能超过物理核（2 个 Vector 核和 1 个 Cube 核组合为 1 个物理核）的核数。
    - AIC/AIV 的核数分别通过 `GetCoreNumAic` 和 `GetCoreNumAiv` 接口获取。
- 在设置 Device 资源限制的场景下，设置的 blockDim 核数不能超过通过 `GetCoreNumAiv` 等接口获取的物理核数。例如，如果使用 `aclrtSetStreamResLimit` 设置 Stream 级别的 Device 资源限制为 8 个核，那么 blockDim 不能超过 8，否则会抢占其他 Stream 的资源，导致资源限制失效。
- 如果开发者使用了 Device 资源限制特性，那么算子设置的 blockDim 不应超过 PlatformAscendC 提供核数的 API（`GetCoreNum`/`GetCoreNumAic`/`GetCoreNumAiv` 等）返回的核数。例如，使用设置 Stream 级别的 Vector 核数为 8，那么 `GetCoreNumAiv` 接口返回值为 8，针对 Vector 算子设置的 blockDim 不应超过 8，否则会抢占其他 Stream 的资源，导致资源限制失效。

## 返回值说明
设置成功时返回 `ge::GRAPH_SUCCESS`。

关于 `graphStatus` 的定义，请参见 15.2.3.55 `ge::graphStatus`。

## 约束说明
无。

## 调用示例
```cpp
ge::graphStatus Tiling4XXX(TilingContext* context) {
    auto ret = context->SetBlockDim(32);
    // ...
}
```
