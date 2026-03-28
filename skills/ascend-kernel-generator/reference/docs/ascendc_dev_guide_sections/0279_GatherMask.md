###### GatherMask

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | ✓ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | ✓ |

| 产品 | 是否支持 |
|------|----------|
| Atlas 200I/500 A2 推理产品 | ✓ |
| Atlas 推理系列产品AI Core | ✓ |
| Atlas 推理系列产品Vector Core | ✗ |
| Atlas 训练系列产品 | ✗ |

## 功能说明

以内置固定模式对应的二进制或者用户自定义输入的Tensor数值对应的二进制为gather mask（数据收集的掩码），从源操作数中选取元素写入目的操作数中。

## 函数原型

### 用户自定义模式

```cpp
template <typename T, typename U, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<U>& src1Pattern, const bool reduceMode, const uint32_t mask, const GatherMaskParams& gatherMaskParams, uint64_t& rsvdCnt)
```

### 内置固定模式

```cpp
template <typename T, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMask(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const uint8_t src1Pattern, const bool reduceMode, const uint32_t mask, const GatherMaskParams& gatherMaskParams, uint64_t& rsvdCnt)
```

## 参数说明

### 模板参数说明

| 参数名称 | 含义 |
|----------|------|
| T | 源操作数src0和目的操作数dst的数据类型。<br>Atlas 推理系列产品AI Core，支持的数据类型为：half/uint16_t/int16_t/float/uint32_t/int32_t<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/bfloat16_t/uint16_t/int16_t/float/uint32_t/int32_t<br>Atlas 200I/500 A2 推理产品，支持的数据类型为：half/uint16_t/int16_t/float/uint32_t/int32_t<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/bfloat16_t/uint16_t/int16_t/float/uint32_t/int32_t |
| U | 用户自定义模式下src1Pattern的数据类型。支持的数据类型为uint16_t/uint32_t。<br>• 当目的操作数数据类型为half/uint16_t/int16_t时，src1Pattern应为uint16_t数据类型。<br>• 当目的操作数数据类型为float/uint32_t/int32_t时，src1Pattern应为uint32_t数据类型。 |
| mode | 预留参数，为后续功能做预留，当前提供默认值，用户无需设置该参数。 |

### 参数说明

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| dst | 输出 | 目的操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>LocalTensor的起始地址需要32字节对齐。 |
| src0 | 输入 | 源操作数。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。<br>LocalTensor的起始地址需要32字节对齐。<br>数据类型需要与目的操作数保持一致。 |
| src1Pattern | 输入 | gather mask（数据收集的掩码），分为内置固定模式和用户自定义模式两种，根据内置固定模式对应的二进制或者用户自定义输入的Tensor数值对应的二进制从源操作数中选取元素写入目的操作数中。1为选取，0为不选取。<br><br>**内置固定模式**：src1Pattern数据类型为uint8_t，取值范围为[1,7]，所有repeat迭代使用相同的gather mask。不支持配置src1RepeatStride。<br>• 1：01010101…0101 # 每个repeat取偶数索引元素<br>• 2：10101010…1010 # 每个repeat取奇数索引元素<br>• 3：00010001…0001 # 每个repeat内每四个元素取第一个元素<br>• 4：00100010…0010 # 每个repeat内每四个元素取第二个元素<br>• 5：01000100…0100 # 每个repeat内每四个元素取第三个元素<br>• 6：10001000…1000 # 每个repeat内每四个元素取第四个元素<br>• 7：11111111...1111 # 每个repeat内取全部元素<br><br>支持情况：<br>• Atlas 推理系列产品AI Core支持模式1-6<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品支持模式1-7<br>• Atlas A3 训练系列产品/Atlas A3 推理系列产品支持模式1-7<br>• Atlas 200I/500 A2 推理产品支持模式1-7<br><br>**用户自定义模式**：src1Pattern数据类型为LocalTensor，迭代间间隔由src1RepeatStride决定，迭代内src1Pattern连续消耗。 |
| reduceMode | 输入 | 用于选择mask参数模式，数据类型为bool，支持如下取值。<br><br>**false**：Normal模式。该模式下，每次repeat操作256Bytes数据，总的数据计算量为repeatTimes * 256Bytes。<br>• mask参数无效，建议设置为0。<br>• 按需配置repeatTimes、src0BlockStride、src0RepeatStride参数。<br>• 支持src1Pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1RepeatStride。<br><br>**true**：Counter模式。根据mask等参数含义的不同，该模式有以下两种配置方式：<br><br>**配置方式一**：每次repeat操作mask个元素，总的数据计算量为repeatTimes * mask个元素。<br>• mask值配置为每一次repeat计算的元素个数。<br>• 按需配置repeatTimes、src0BlockStride、src0RepeatStride参数。<br>• 支持src1Pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1RepeatStride。<br><br>**配置方式二**：总的数据计算量为mask个元素。<br>• mask配置为总的数据计算量。<br>• repeatTimes值不生效，指令的迭代次数由源操作数和mask共同决定。<br>• 按需配置src0BlockStride、src0RepeatStride参数。<br>• 支持src1Pattern配置为内置固定模式或用户自定义模式。用户自定义模式下可根据实际情况配置src1RepeatStride。<br><br>支持情况：<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持配置方式一<br>• Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持配置方式一<br>• Atlas 200I/500 A2 推理产品，支持配置方式一<br>• Atlas 推理系列产品AI Core，支持配置方式二 |
| mask | 输入 | 用于控制每次迭代内参与计算的元素。根据reduceMode，分为两种模式：<br>• Normal模式：mask无效，建议设置为0。<br>• Counter模式：取值范围[1, 2³² – 1]。不同的版本型号Counter模式下，mask参数表示含义不同。具体配置规则参考上文reduceMode参数描述。 |
| gatherMaskParams | 输入 | 控制操作数地址步长的数据结构，GatherMaskParams类型。<br>具体定义请参考`${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_gather.h`，`${INSTALL_DIR}`请替换为CANN软件安装后文件存储路径。<br>具体参数说明见表15-172。 |
| rsvdCnt | 输出 | 该条指令筛选后保留下来的元素计数，对应dstLocal中有效元素个数，数据类型为uint64_t。 |

### GatherMaskParams 结构体参数说明

| 参数名称 | 含义 |
|----------|------|
| src0BlockStride | 用于设置src0同一迭代不同DataBlock间的地址步长（起始地址之间的间隔）。单位为DataBlock。 |
| repeatTimes | 迭代次数。 |
| src0RepeatStride | 用于设置src0相邻迭代间的地址步长（起始地址之间的间隔）。单位为DataBlock。 |
| src1RepeatStride | 用于设置src1相邻迭代间的地址步长（起始地址之间的间隔）。单位为DataBlock。 |

## 返回值说明

无

## 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- 操作数地址重叠约束请参考通用地址重叠约束。
- 若调用该接口前为Counter模式，在调用该接口后需要显式设置回Counter模式（接口内部执行结束后会设置为Normal模式）。

## 调用示例

### 用户自定义Tensor样例

```cpp
#include "kernel_operator.h"

class KernelGatherMask {
public:
    __aicore__ inline KernelGatherMask() {}
    
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ uint32_t*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ uint32_t*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ uint32_t*)dstGm);
        
        pipe.InitBuffer(inQueueSrc0, 1, 256 * sizeof(uint32_t));
        pipe.InitBuffer(inQueueSrc1, 1, 32 * sizeof(uint32_t));
        pipe.InitBuffer(outQueueDst, 1, 256 * sizeof(uint32_t));
    }
    
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
    
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<uint32_t> src0Local = inQueueSrc0.AllocTensor<uint32_t>();
        AscendC::LocalTensor<uint32_t> src1Local = inQueueSrc1.AllocTensor<uint32_t>();
        AscendC::DataCopy(src0Local, src0Global, 256);
        AscendC::DataCopy(src1Local, src1Global, 32);
        inQueueSrc0.EnQue(src0Local);
        inQueueSrc1.EnQue(src1Local);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<uint32_t> src0Local = inQueueSrc0.DeQue<uint32_t>();
        AscendC::LocalTensor<uint32_t> src1Local = inQueueSrc1.DeQue<uint32_t>();
        AscendC::LocalTensor<uint32_t> dstLocal = outQueueDst.AllocTensor<uint32_t>();
        
        uint32_t mask = 70;
        uint64_t rsvdCnt = 0;
        // reduceMode = true; 使用Counter模式
        // src0BlockStride = 1; 单次迭代内数据间隔1个datablock，即数据连续读取和写入
        // repeatTimes = 2; Counter模式时，仅在部分产品型号下会生效
        // src0RepeatStride = 4; 源操作数迭代间数据间隔4个datablock
        // src1RepeatStride = 0; src1迭代间数据间隔0个datablock，即原位置读取
        AscendC::GatherMask(dstLocal, src0Local, src1Local, true, mask, { 1, 2, 4, 0 }, rsvdCnt);
        
        outQueueDst.EnQue<uint32_t>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
        inQueueSrc1.FreeTensor(src1Local);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<uint32_t> dstLocal = outQueueDst.DeQue<uint32_t>();
        AscendC::DataCopy(dstGlobal, dstLocal, 256);
        outQueueDst.FreeTensor(dstLocal);
    }
    
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc0, inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<uint32_t> src0Global, src1Global, dstGlobal;
};

extern "C" __global__ __aicore__ void gather_mask_simple_kernel(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    KernelGatherMask op;
    op.Init(src0Gm, src1Gm, dstGm);
    op.Process();
}
```

**Counter模式配置方式一示意图**：
- mask = 70，每一次repeat计算70个元素；
- repeatTimes = 2，共进行2次repeat；
- src0BlockStride = 1，源操作数src0Local单次迭代内datablock之间无间隔；
- src0RepeatStride = 4，源操作数src0Local相邻迭代间的间隔为4个datablock，所以第二次repeat从第33个元素开始处理。
- src1Pattern配置为用户自定义模式。src1RepeatStride = 0，src1Pattern相邻迭代间的间隔为0个datablock，所以第二次repeat仍从src1Pattern的首地址开始处理。

**Counter模式配置方式二示意图**：
- mask = 70，一共计算70个元素；
- repeatTimes配置不生效，根据源操作数和mask自动推断：源操作数的数据类型为uint32_t，每个迭代处理256Bytes数据，一个迭代处理64个元素，共需要进行2次repeat；
- src0BlockStride = 1，源操作数src0Local单次迭代内datablock之间无间隔；
- src0RepeatStride = 4，源操作数src0Local相邻迭代间的间隔为4个datablock，所以第二次repeat从第33个元素开始处理。
- src1Pattern配置为用户自定义模式。src1RepeatStride = 0，src1Pattern相邻迭代间的间隔为0个datablock，所以第二次repeat仍从src1Pattern的首地址开始处理。

### 内置固定模式

```cpp
#include "kernel_operator.h"

class KernelGatherMask {
public:
    __aicore__ inline KernelGatherMask() {}
    
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ uint16_t*)src0Gm);
        dstGlobal.SetGlobalBuffer((__gm__ uint16_t*)dstGm);
        pipe.InitBuffer(inQueueSrc0, 1, 128 * sizeof(uint16_t));
        pipe.InitBuffer(outQueueDst, 1, 128 * sizeof(uint16_t));
    }
    
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
    
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<uint16_t> src0Local = inQueueSrc0.AllocTensor<uint16_t>();
        AscendC::DataCopy(src0Local, src0Global, 128);
        inQueueSrc0.EnQue(src0Local);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<uint16_t> src0Local = inQueueSrc0.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> dstLocal = outQueueDst.AllocTensor<uint16_t>();
        
        uint32_t mask = 0; // normal模式下mask建议设置为0
        uint64_t rsvdCnt = 0; // 用于保存筛选后保留下来的元素个数
        uint8_t src1Pattern = 2; // 内置固定模式
        // reduceMode = false; 使用normal模式
        // src0BlockStride = 1; 单次迭代内数据间隔1个Block，即数据连续读取和写入
        // repeatTimes = 1;重复迭代一次
        // src0RepeatStride = 0;重复一次，故设置为0
        // src1RepeatStride = 0;重复一次，故设置为0
        AscendC::GatherMask(dstLocal, src0Local, src1Pattern, false, mask, { 1, 1, 0, 0 }, rsvdCnt);
        
        outQueueDst.EnQue<uint16_t>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<uint16_t> dstLocal = outQueueDst.DeQue<uint16_t>();
        AscendC::DataCopy(dstGlobal, dstLocal, 128);
        outQueueDst.FreeTensor(dstLocal);
    }
    
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc0;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<uint16_t> src0Global, dstGlobal;
};

extern "C" __global__ __aicore__ void gather_mask_simple_kernel(__gm__ uint8_t* src0Gm, __gm__ uint8_t* dstGm)
{
    KernelGatherMask op;
    op.Init(src0Gm, dstGm);
    op.Process();
}
```

**结果示例如下**：

输入数据src0Local：`[1 2 3 ... 128]`

输入数据src1Pattern：`src1Pattern = 2`

输出数据dstLocal：`[2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100 102 104 106 108 110 112 114 116 118 120 122 124 126 128 undefine ..undefined]`

输出数据rsvdCnt：`64`
