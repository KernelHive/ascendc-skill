##### GetStoreAtomicConfig(ISASI)

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

获取原子操作使能位与原子操作类型的值，详细说明见表15-420。

## 函数原型

```cpp
__aicore__ inline void GetStoreAtomicConfig(uint16_t &atomicType, uint16_t &atomicOp)
```

## 参数说明

**表 15-421 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| atomicType | 输出 | 原子操作使能位。<br>0：无原子操作<br>1：使能原子操作，进行原子操作的数据类型为float<br>2：使能原子操作，进行原子操作的数据类型为half<br>3：使能原子操作，进行原子操作的数据类型为int16_t<br>4：使能原子操作，进行原子操作的数据类型为int32_t<br>5：使能原子操作，进行原子操作的数据类型为int8_t<br>6：使能原子操作，进行原子操作的数据类型为bfloat16_t |
| atomicOp | 输出 | 原子操作类型。<br>0：求和操作 |

## 返回值说明

无

## 约束说明

此接口需要与15.1.4.7.6 SetStoreAtomicConfig(ISASI)配合使用，用以获取原子操作使能位与原子操作类型的值。

## 调用示例

```cpp
AscendC::SetStoreAtomicConfig<AscendC::AtomicDtype::ATOMIC_F16, AscendC::AtomicOp::ATOMIC_SUM>();
uint16_t type = 0; // 原子操作使能位
uint16_t op = 0; // 原子操作类型
AscendC::GetStoreAtomicConfig(type, op);
```
