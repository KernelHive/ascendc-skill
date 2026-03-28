#### 矩阵乘输出的 Channel 拆分

## 功能介绍

矩阵乘输出的 Channel 拆分，又称 ChannelSplit。指当 Matmul 计算结果 C 矩阵的格式为 NZ 时，C 矩阵采用分形存储。关于 NZ 格式的详细内容请参考数据格式。

当 C 矩阵的物理排布格式为 NZ、数据类型为 float 时，默认情况下，每个分形内部包含 16×16 个元素，即分形的大小为 16×16。ChannelSplit 的功能为将此场景下 C 矩阵的每个 16×16 的分形切分为 16×8 的分形，使得 C 矩阵按照 16×8 的分形进行存储。

由于 1 个 float 类型数据的大小为 4 字节，16×8 的分形在内轴满足 32 字节对齐，内轴上的数据量与一条 NPU 矢量计算指令处理的数据单元一致，这便于后续的其它计算。

ChannelSplit 功能默认不启用，用户需通过设置 MatmulConfig 中的 `isEnableChannelSplit` 参数为 `true` 来开启此功能。

图 6-32 ChannelSplit 功能示意图

## 使用场景

对于 NZ 格式、float 类型的 C 矩阵，需要按 16×8 的分形存储时，使用该功能。

## 约束说明

开启 ChannelSplit 功能需满足：

- C 矩阵的数据排布格式为 `CubeFormat::NZ`
- C 矩阵的数据类型为 `float`
- C 矩阵的内存逻辑位置为 Global Memory

## 调用示例

完整的算子样例请参考 matmul_channelsplit 算子样例。

```cpp
// 指定获取和修改的 MatmulConfig 模板
constexpr static MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
// 修改模板参数 isEnableChannelSplit=true，开启该 MatmulConfig 模板的 ChannelSplit 功能
constexpr static MatmulFuncParams funcParamsChannelSplit{
    false, false, false, false, 0, IterateOrder::ORDER_M, ScheduleType::INNER_PRODUCT, true, false, false,
    false, true /*isEnableChannelSplit*/
};
constexpr static MatmulConfig MM_CFG = GetMMConfig<configMode>(funcParamsChannelSplit);
Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;

// 常规 Matmul 计算，最后输出分形为 16×8
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
