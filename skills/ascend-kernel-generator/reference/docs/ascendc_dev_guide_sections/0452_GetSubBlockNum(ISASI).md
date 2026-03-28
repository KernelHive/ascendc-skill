##### GetSubBlockNum(ISASI)

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

分离模式下，获取一个AI Core上Cube Core（AIC）或者Vector Core（AIV）的数量。

## 函数原型

```cpp
__aicore__ inline int64_t GetSubBlockNum()
```

## 参数说明

无

## 返回值说明

不同Kernel类型下（通过15.1.4.10.10 设置Kernel类型设置），在AIC和AIV上调用该接口的返回值如下：

**表 15-413 返回值列表**

| Kernel 类型 | KERNEL_TYPE_AIV_ONLY | KERNEL_TYPE_AIC_ONLY | KERNEL_TYPE_MIX_AIC_1_2 | KERNEL_TYPE_MIX_AIC_1_1 | KERNEL_TYPE_MIX_AIC_1_0 | KERNEL_TYPE_MIX_AIV_1_0 |
|-------------|----------------------|----------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| AIV         | 1                    | -                    | 2                       | 1                       | -                       | 1                       |
| AIC         | -                    | 1                    | 1                       | 1                       | 1                       | -                       |

## 约束说明

无

## 调用示例

```cpp
int64_t subBlockNum = AscendC::GetSubBlockNum();
```
