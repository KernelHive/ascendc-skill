###### IterateAll

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

调用一次 IterateAll，会计算出 `singleCoreM * singleCoreN` 大小的 C 矩阵。迭代顺序可通过 tiling 参数 `iterateOrder` 调整。

## 函数原型

```cpp
template <bool sync = true> 
__aicore__ inline void IterateAll(const GlobalTensor<DstT>& gm, 
                                 uint8_t enAtomic = 0, 
                                 bool enSequentialWrite = false, 
                                 bool waitIterateAll = false, 
                                 bool fakeMsg = false)

template <bool sync = true> 
__aicore__ inline void IterateAll(const LocalTensor<DstT>& ubCmatrix, 
                                 uint8_t enAtomic = 0)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| sync | 获取 C 矩阵过程分为同步和异步两种模式：<br>● 同步：需要同步等待 IterateAll 执行结束<br>● 异步：不需要同步等待 IterateAll 执行结束<br>通过该参数设置同步或者异步模式：同步模式设置为 true；异步模式设置为 false，默认为同步模式。<br>Atlas 200I/500 A2 推理产品只支持设置为 true。 |

### 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| gm | 输出 | C 矩阵。类型为 GlobalTensor。<br>**Atlas A3 训练系列产品/Atlas A3 推理系列产品**，支持的数据类型为：half/float/bfloat16_t/int32_t/int8_t<br>**Atlas 推理系列产品AI Core**，支持的数据类型为：half/float/int8_t/int32_t<br>**Atlas A2 训练系列产品/Atlas A2 推理系列产品**，支持的数据类型为：half/float/bfloat16_t/int32_t/int8_t<br>**Atlas 200I/500 A2 推理产品**，支持的数据类型为 half/float/bfloat16_t/int32_t |
| ubCmatrix | 输出 | C 矩阵。类型为 LocalTensor，支持的 TPosition 为 TSCM。<br>**Atlas A3 训练系列产品/Atlas A3 推理系列产品**，支持的数据类型为：half/float/bfloat16_t/int32_t/int8_t<br>**Atlas 推理系列产品AI Core**不支持包含该参数的原型接口<br>**Atlas A2 训练系列产品/Atlas A2 推理系列产品**，支持的数据类型为：half/float/bfloat16_t/int32_t/int8_t<br>**Atlas 200I/500 A2 推理产品**，支持的数据类型为 half/float/bfloat16_t/int32_t |
| enAtomic | 输入 | 是否开启 Atomic 操作，默认值为 0。<br>参数取值：<br>● 0：不开启 Atomic 操作<br>● 1：开启 AtomicAdd 累加操作<br>● 2：开启 AtomicMax 求最大值操作<br>● 3：开启 AtomicMin 求最小值操作<br>对于 **Atlas 推理系列产品AI Core**，只有输出位置是 GM 才支持开启 Atomic 操作。<br>对于 **Atlas 200I/500 A2 推理产品**，只有输出位置是 GM 才支持开启 Atomic 操作。 |
| enSequentialWrite | 输入 | 是否开启连续写模式（连续写，写入 [baseM, baseN]；非连续写，写入 [singleCoreM, singleCoreN] 中对应的位置），默认值 false（非连续写模式）。<br>Atlas 200I/500 A2 推理产品不支持该参数。 |
| waitIterateAll | 输入 | 仅在异步场景下使用，是否需要通过 WaitIterateAll 接口等待 IterateAll 执行结束。<br>● true：需要通过 WaitIterateAll 接口等待 IterateAll 执行结束<br>● false：不需要通过 WaitIterateAll 接口等待 IterateAll 执行结束，开发者自行处理等待 IterateAll 执行结束的过程 |
| fakeMsg | 输入 | 仅在 IBShare 场景（模板参数中开启了 doIBShareNorm 开关）和 IntraBlockPartSum 场景（模板参数中开启了 intraBlockPartSum 开关）使用。<br>● **IBShare 场景**：该场景复用 L1 上相同的 A 矩阵或 B 矩阵数据，要求 AIV 分核调用 IterateAll 的次数必须匹配，此时需要调用 IterateAll 并设置 fakeMsg 为 true，不执行真正的计算，仅用来保证 IterateAll 调用成对出现。默认值为 false，表示执行真正的计算。<br>● **IntraBlockPartSum 场景**：用于分离模式下的 Vector 计算、Cube 计算融合，实现多个 AIV 核的一次 Matmul 计算结果（baseM * baseN 大小的矩阵分片）在 L0C Buffer 上累加。默认值为 false，表示执行各 AIV 核的 Matmul 计算结果在 L0C Buffer 上累加。 |

## 返回值说明

无

## 约束说明

传入的 C 矩阵地址空间大小需要保证不小于 `singleCoreM * singleCoreN` 个元素。

## 调用示例

IterateAll 接口的调用示例如下，更多异步场景的算子样例请参考 IterateAll 异步场景矩阵乘法。

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c); // 计算
```
