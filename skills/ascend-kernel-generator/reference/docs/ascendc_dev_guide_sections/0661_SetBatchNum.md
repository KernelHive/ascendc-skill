###### SetBatchNum

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

| 产品 | 是否支持 |
|------|----------|
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

在不改变Tiling的情况下，重新设置多Batch计算的Batch数。

## 函数原型

```cpp
__aicore__ inline void SetBatchNum(int32_t batchA, int32_t batchB)
```

## 参数说明

**表 15-627 参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| batchA | 输入 | 设置的一次计算的A矩阵Batch数 |
| batchB | 输入 | 设置的一次计算的B矩阵Batch数 |

## 返回值说明

无

## 约束说明

当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

## 调用示例

```cpp
// 纯cube模式
mm1.SetTensorA(gm_a, isTransposeAIn);
mm1.SetTensorB(gm_b, isTransposeBIn);
if(tiling.isBias) {
    mm1.SetBias(gm_bias);
}
mm1.SetBatchNum(batchA, batchB);
// 多batch Matmul计算
mm1.IterateBatch(gm_c, false, 0, false);
```
