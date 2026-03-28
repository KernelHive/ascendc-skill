##### LocalTensor 构造函数

## 产品支持情况

| 产品 | 是否支持（Pipe 框架） | 是否支持（更底层编程） |
|------|----------------------|----------------------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ | √ |
| Atlas 200I/500 A2 推理产品 | √ | × |
| Atlas 推理系列产品 AI Core | √ | √ |
| Atlas 推理系列产品 Vector Core | √ | √ |
| Atlas 训练系列产品 | √ | × |

## 功能说明

LocalTensor 构造函数。

## 函数原型

- 适用于 Pipe 编程框架，通常情况下开发者不直接调用，该函数不会对 LocalTensor 成员变量赋初值，均为随机值。

```cpp
__aicore__ inline LocalTensor<T>() {}
```

- 适用于更底层编程，根据指定的逻辑位置/地址/长度，返回 Tensor 对象。

```cpp
__aicore__ inline LocalTensor<T>(TPosition pos, uint32_t addr, uint32_t tileSize)
__aicore__ inline LocalTensor<T>(uint32_t addr)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | - 适用于 Pipe 编程框架的原型，支持基础数据类型以及 TensorTrait 类型。<br>- 适用于更底层编程的原型，支持的数据类型如下：<br>  - `__aicore__ inline LocalTensor<T>(TPosition pos, uint32_t addr, uint32_t tileSize)`：仅支持基础数据类型<br>  - `__aicore__ inline LocalTensor<T>(uint32_t addr)`：仅支持 TensorTrait 类型 |

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| pos | 输入 | LocalTensor 所在的逻辑位置 |
| addr | 输入 | LocalTensor 的起始地址，其范围为 [0, 对应物理内存最大值)。起始地址需要保证 32 字节对齐 |
| tileSize | 输入 | LocalTensor 的元素个数，addr 和 tileSize（转换成所占字节数）之和不应超出对应物理内存的范围 |

## 返回值说明

无

## 约束说明

无

## 调用示例

本节提供了 LocalTensor 构造函数的使用示例和其所有成员函数的调用示例。

### 示例 1
```cpp
// srcLen = 256, num = 100, M=50
for (int32_t i = 0; i < srcLen; ++i) {
    inputLocal.SetValue(i, num); // 对 inputLocal 中第 i 个位置进行赋值为 num
}
// 示例1结果如下：
// 数据(inputLocal): [100 100 100 ... 100]
```

### 示例 2
```cpp
for (int32_t i = 0; i < srcLen; ++i) {
    auto element = inputLocal.GetValue(i); // 获取 inputLocal 中第 i 个位置的数值
}
// 示例2结果如下：
// element 为 100
```

### 示例 3
```cpp
for (int32_t i = 0; i < srcLen; ++i) {
    inputLocal(i) = num; // 对 inputLocal 中第 i 个位置进行赋值为 num
}
// 示例3结果如下：
// 数据(inputLocal): [100 100 100 ... 100]
```

### 示例 4
```cpp
for (int32_t i = 0; i < srcLen; ++i) {
    auto element = inputLocal(i); // 获取 inputLocal 中第 i 个位置的数值
}
// 示例4结果如下：
// element 为 100
```

### 示例 5
```cpp
auto size = inputLocal.GetSize(); // 获取 inputLocal 的长度，size 大小为 inputLocal 有多少个元素
// 示例5结果如下：
// size 大小为 srcLen，256
```

### 示例 6
```cpp
// operator[] 使用方法, inputLocal[16] 为从起始地址开始偏移量为 16 的新 tensor
AscendC::Add(outputLocal[16], inputLocal[16], inputLocal2[16], M);
// 示例6结果如下：
// 输入数据(inputLocal): [100 100 100 ... 100]
// 输入数据(inputLocal2): [1 2 3 ... 66]
// 输出数据(outputLocal): [... 117 118 119 ... 166]
```

### 示例 7
```cpp
AscendC::TTagType tag = 10;
inputLocal.SetUserTag(tag); // 对 LocalTensor 设置 tag 信息
```

### 示例 8
```cpp
AscendC::LocalTensor<half> tensor1 = que1.DeQue<half>();
AscendC::TTagType tag1 = tensor1.GetUserTag();
AscendC::LocalTensor<half> tensor2 = que2.DeQue<half>();
AscendC::TTagType tag2 = tensor2.GetUserTag();
AscendC::LocalTensor<half> tensor3 = que3.AllocTensor<half>();
/* 使用 Tag 控制条件语句执行 */
if ((tag1 <= 10) && (tag2 >= 9)) {
    AscendC::Add(tensor3, tensor1, tensor2, TILE_LENGTH); // 当 tag1 小于等于 10，tag2 大于等于 9 的时候，才能进行相加操作
}
```

### 示例 9
```cpp
// input_local 为 int32_t 类型，包含 16 个元素(64 字节)
for (int32_t i = 0; i < 16; ++i) {
    inputLocal.SetValue(i, i); // 对 inputLocal 中第 i 个位置进行赋值为 i
}

// 调用 ReinterpretCast 将 input_local 重解释为 int16_t 类型
AscendC::LocalTensor<int16_t> interpreTensor = inputLocal.ReinterpretCast<int16_t>();
// 示例9结果如下，二者数据完全一致，在物理内存上也是同一地址，仅根据不同类型进行了重解释
// inputLocal: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
// interpreTensor: 0 0 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 9 0 10 0 11 0 12 0 13 0 14 0 15 0
```

### 示例 10
```cpp
// 调用 GetPhyAddr() 返回 LocalTensor 地址，CPU 上返回的是指针类型(T*)，NPU 上返回的是物理存储的地址(uint64_t)
#ifdef ASCEND_CPU_DEBUG
float *inputLocalCpuPtr = inputLocal.GetPhyAddr();
uint64_t realAddr = (uint64_t)inputLocalCpuPtr - (uint64_t)(GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(AscendC::TPosition::VECCALC)));
#else
uint64_t realAddr = inputLocal.GetPhyAddr();
#endif
```

### 示例 11
```cpp
AscendC::TPosition srcPos = (AscendC::TPosition)inputLocal.GetPosition();
if (srcPos == AscendC::TPosition::VECCALC) {
    // 处理逻辑 1
} else if (srcPos == AscendC::TPosition::A1) {
    // 处理逻辑 2
} else {
    // 处理逻辑 3
}
```

### 示例 12
```cpp
// 获取 localTensor 的长度(单位为 Byte)，数据类型为 int32_t，所以是 16*sizeof(int32_t)
uint32_t len = inputLocal.GetLength();
// inputLocal: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
// len: 64
```

### 示例 13
```cpp
// 设置 Tensor 的 ShapeInfo 信息
AscendC::LocalTensor<float> maxUb = softmaxMaxBuf.template Get<float>();
uint32_t shapeArray[] = {16, 1024};
maxUb.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
```

### 示例 14
```cpp
// 获取 Tensor 的 ShapeInfo 信息
AscendC::ShapeInfo maxShapeInfo = maxUb.GetShapeInfo();
uint32_t orgShape0 = maxShapeInfo.originalShape[0];
uint32_t orgShape1 = maxShapeInfo.originalShape[1];
uint32_t orgShape2 = maxShapeInfo.originalShape[2];
uint32_t orgShape3 = maxShapeInfo.originalShape[3];
uint32_t shape2 = maxShapeInfo.shape[2];
```

### 示例 15
```cpp
// SetAddrWithOffset，用于快速获取定义一个 Tensor，同时指定新 Tensor 相对于旧 Tensor 首地址的偏移
// 需要注意，偏移的长度为旧 Tensor 的元素个数
AscendC::LocalTensor<float> tmpBuffer1 = tempBmm2Queue.AllocTensor<float>();
AscendC::LocalTensor<half> tmpHalfBuffer;
tmpHalfBuffer.SetAddrWithOffset(tmpBuffer1, calcSize * 2);
```

### 示例 16
```cpp
// SetBufferLen 如下示例将申请的 Tensor 长度修改为 1024(单位为字节)
AscendC::LocalTensor<float> tmpBuffer2 = tempBmm2Queue.AllocTensor<float>();
tmpBuffer2.SetBufferLen(1024);
```

### 示例 17
```cpp
// SetSize 如下示例将申请的 Tensor 长度修改为 256(单位为元素)
AscendC::LocalTensor<float> tmpBuffer3 = tempBmm2Queue.AllocTensor<float>();
tmpBuffer3.SetSize(256);
```

### 示例 18
```cpp
#ifdef ASCEND_CPU_DEBUG
// 只限于 CPU 调试，将 LocalTensor 数据 Dump 到文件中，用于精度调试，文件保存在执行目录
AscendC::LocalTensor<float> tmpTensor = softmaxMaxBuf.template Get<float>();
tmpTensor.ToFile("tmpTensor.bin");
#endif
```

### 示例 19
```cpp
#ifdef ASCEND_CPU_DEBUG
// 只限于 CPU 调试，在调试窗口中打印 LocalTensor 数据用于精度调试，每一行打印一个 datablock(32Bytes) 的数据
AscendC::LocalTensor<int32_t> inputLocal = softmaxMaxBuf.template Get<int32_t>();
for (int32_t i = 0; i < 16; ++i) {
    inputLocal.SetValue(i, i); // 对 input_local 中第 i 个位置进行赋值为 i
}
inputLocal.Print();
// 0000: 0 1 2 3 4 5 6 7 8
// 0008: 9 10 11 12 13 14 15
#endif
```

### 示例 20
```cpp
// 在更底层编程场景使用，根据传入的逻辑位置 VECIN、起始地址 128、元素个数 32、数据类型 float，构造出 Tensor 对象
uint32_t addr = 128;
uint32_t tileSize = 32;
AscendC::LocalTensor<float> tensor1 = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, addr, tileSize);

// 根据传入的 TensorTrait 信息、起始地址 128 构造出 Tensor 对象
// 其逻辑位置为 VECIN，数据类型为 float，Tensor 元素个数为 16*16*16
template <uint32_t v>
using UIntImm = Std::integral_constant<uint32_t, v>;
...
auto shape = AscendC::MakeShape(UIntImm<16>{}, UIntImm<16>{}, UIntImm<16>{});
auto stride = AscendC::MakeStride(UIntImm<0>{}, UIntImm<0>{}, UIntImm<0>{});
auto layoutMake = AscendC::MakeLayout(shape, stride);
auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);

uint32_t addr = 128;
auto tensor1 = AscendC::LocalTensor<decltype(tensorTraitMake)>(addr);
```
