###### InitBufPool

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | √ |

## 功能说明

初始化 TBufPool 内存资源池。本接口适用于内存资源有限时，希望手动指定 UB/L1 内存资源复用的场景。本接口初始化后在整体内存资源中划分出一块子资源池。划分出的子资源池 TBufPool，提供了如下方式进行资源管理：

- `TPipe::InitBufPool` 的重载接口指定与其他 TBufPool 子资源池复用
- `TBufPool::InitBufPool` 接口对子资源池继续划分
- `TBufPool::InitBuffer` 接口分配 Buffer

关于 TBufPool 的具体介绍及资源划分图示请参考 15.1.4.4.3 TBufPool。

## 函数原型

```cpp
template <class T>
__aicore__ inline bool InitBufPool(T& bufPool, uint32_t len)

template <class T, class U>
__aicore__ inline bool InitBufPool(T& bufPool, uint32_t len, U& shareBuf)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | bufPool 的类型 |
| U | shareBuf 的类型 |

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| bufPool | 输入 | 新划分的资源池，类型为 TBufPool |
| len | 输入 | 新划分资源池长度，单位为 Byte，非 32Bytes 对齐会自动补齐至 32Bytes 对齐 |
| shareBuf | 输入 | 被复用资源池，类型为 TBufPool，新划分资源池与被复用资源池共享起始地址及长度 |

## 约束说明

- 新划分的资源池与被复用资源池的硬件属性需要一致，两者共享起始地址及长度
- 输入长度需要小于等于被复用资源池长度
- 其他泛用约束参考 15.1.4.4.3 TBufPool

## 返回值说明

无

## 调用示例

由于物理内存的大小有限，在计算过程没有数据依赖的场景或数据依赖串行的场景下，可以通过指定内存复用解决资源不足的问题。本示例中 `Tpipe::InitBufPool` 初始化子资源池 `tbufPool1`，并且指定 `tbufPool2` 复用 `tbufPool1` 的起始地址及长度；`tbufPool1` 及 `tbufPool2` 的后续计算串行，不存在数据踩踏，实现了内存复用及自动同步的能力。

```cpp
#include "kernel_operator.h"

class ResetApi {
public:
    __aicore__ inline ResetApi() {}
    
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBufPool(tbufPool1, 196608);
        pipe.InitBufPool(tbufPool2, 196608, tbufPool1);
    }
    
    __aicore__ inline void Process()
    {
        tbufPool1.InitBuffer(queSrc0, 1, 65536);
        tbufPool1.InitBuffer(queSrc1, 1, 65536);
        tbufPool1.InitBuffer(queDst0, 1, 65536);
        CopyIn();
        Compute();
        CopyOut();
        tbufPool1.Reset();
        
        tbufPool2.InitBuffer(queSrc2, 1, 65536);
        tbufPool2.InitBuffer(queSrc3, 1, 65536);
        tbufPool2.InitBuffer(queDst1, 1, 65536);
        CopyIn1();
        Compute1();
        CopyOut1();
        tbufPool2.Reset();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> src0Local = queSrc0.AllocTensor<half>();
        AscendC::LocalTensor<half> src1Local = queSrc1.AllocTensor<half>();
        AscendC::DataCopy(src0Local, src0Global, 512);
        AscendC::DataCopy(src1Local, src1Global, 512);
        queSrc0.EnQue(src0Local);
        queSrc1.EnQue(src1Local);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> src0Local = queSrc0.DeQue<half>();
        AscendC::LocalTensor<half> src1Local = queSrc1.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = queDst0.AllocTensor<half>();
        AscendC::Add(dstLocal, src0Local, src1Local, 512);
        queDst0.EnQue<half>(dstLocal);
        queSrc0.FreeTensor(src0Local);
        queSrc1.FreeTensor(src1Local);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = queDst0.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, 512);
        queDst0.FreeTensor(dstLocal);
    }
    
    __aicore__ inline void CopyIn1()
    {
        AscendC::LocalTensor<half> src0Local = queSrc2.AllocTensor<half>();
        AscendC::LocalTensor<half> src1Local = queSrc3.AllocTensor<half>();
        AscendC::DataCopy(src0Local, src0Global, 512);
        AscendC::DataCopy(src1Local, src1Global, 512);
        queSrc2.EnQue(src0Local);
        queSrc3.EnQue(src1Local);
    }
    
    __aicore__ inline void Compute1()
    {
        AscendC::LocalTensor<half> src0Local = queSrc2.DeQue<half>();
        AscendC::LocalTensor<half> src1Local = queSrc3.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = queDst1.AllocTensor<half>();
        AscendC::Add(dstLocal, src0Local, src1Local, 512);
        queDst1.EnQue<half>(dstLocal);
        queSrc2.FreeTensor(src0Local);
        queSrc3.FreeTensor(src1Local);
    }
    
    __aicore__ inline void CopyOut1()
    {
        AscendC::LocalTensor<half> dstLocal = queDst1.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, 512);
        queDst1.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBufPool<AscendC::TPosition::VECCALC> tbufPool1, tbufPool2;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> queSrc0, queSrc1, queSrc2, queSrc3;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> queDst0, queDst1;
    AscendC::GlobalTensor<half> src0Global, src1Global, dstGlobal;
};

extern "C" __global__ __aicore__ void tbufpool_kernel(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm,
                                                      __gm__ uint8_t* dstGm)
{
    ResetApi op;
    op.Init(src0Gm, src1Gm, dstGm);
    op.Process();
}
```
