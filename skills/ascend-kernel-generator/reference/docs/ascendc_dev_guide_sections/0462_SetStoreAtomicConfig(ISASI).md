##### SetStoreAtomicConfig(ISASI)

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

设置原子操作使能位与原子操作类型。

## 函数原型

```cpp
template <AtomicDtype type, AtomicOp op>
__aicore__ inline void SetStoreAtomicConfig()
```

## 参数说明

**表 模板参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| type | 输入 | 原子操作使能位，AtomicDtype枚举类的定义如下：<br>`enum class AtomicDtype {`<br>`ATOMIC_NONE = 0, // 无原子操作`<br>`ATOMIC_F32, // 使能原子操作，进行原子操作的数据类型为float`<br>`ATOMIC_F16, // 使能原子操作，进行原子操作的数据类型为half`<br>`ATOMIC_S16, // 使能原子操作，进行原子操作的数据类型为int16_t`<br>`ATOMIC_S32, // 使能原子操作，进行原子操作的数据类型为int32_t`<br>`ATOMIC_S8, // 使能原子操作，进行原子操作的数据类型为int8_t`<br>`ATOMIC_BF16 // 使能原子操作，进行原子操作的数据类型为bfloat16_t`<br>`};` |
| op | 输入 | 原子操作类型，仅当使能原子操作时有效（即"type"为非"ATOMIC_NONE"的场景），当前仅支持求和操作。<br>`enum class AtomicOp {`<br>`ATOMIC_SUM = 0 // 求和操作`<br>`};` |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
// 设置原子操作为求和操作，支持的数据类型为half
AscendC::SetStoreAtomicConfig<AscendC::AtomicDtype::ATOMIC_F16, AscendC::AtomicOp::ATOMIC_SUM>();
```
