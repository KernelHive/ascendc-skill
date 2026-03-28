##### ArithProgression

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

给定起始值、等差值和长度，返回一个等差数列。

## 实现原理

以float类型、ND格式、firstValue和diffValue输入Scalar为例，描述ArithProgression高阶API内部算法框图，如下图所示。

**图 15-81 ArithProgression 算法框图**

计算过程分为如下几步，均在Vector上进行：

1. **等差数列长度8以内步骤**：按照firstValue和diffValue的值，使用SetValue实现等差数列扩充，扩充长度最大为8，如果等差数列长度小于8，算法结束；
2. **等差数列长度8至64的步骤**：对第一步中的等差数列结果使用Adds进行扩充，最大循环7次扩充至64，如果等差数列长度小于64，算法结束；
3. **等差数列长度64以上的步骤**：对第二步中的等差数列结果使用Adds进行扩充，不断循环直至达到等差数列长度为止。

## 函数原型

```cpp
template <typename T>
__aicore__ inline void ArithProgression(const LocalTensor<T> &dstLocal, const T firstValue, const T diffValue,
const int32_t count)
```

## 参数说明

### 表 15-853 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half/float/int16_t/int32_t<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half/float/int16_t/int32_t<br>Atlas 推理系列产品AI Core，支持的数据类型为：half/float/int16_t/int32_t |

### 表 15-854 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dstLocal | 输出 | 目的操作数。dstLocal的大小应大于等于count * sizeof(T)。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| firstValue | 输入 | 等差数列的首个元素值。 |
| diffValue | 输入 | 等差数列元素之间的差值，应大于等于0。 |
| count | 输入 | 等差数列的长度。count>0。 |

## 返回值说明

无

## 约束说明

当前仅支持ND格式的输入，不支持其他格式。

## 调用示例

完整算子样例请参考arithprogression算子样例。

```cpp
AscendC::LocalTensor<T> dstLocal = outDst.AllocTensor<T>();
AscendC::ArithProgression<T>(dstLocal, static_cast<T>(firstValue_), static_cast<T>(diffValue_), count_);
outDst.EnQue<T>(dstLocal);
```
