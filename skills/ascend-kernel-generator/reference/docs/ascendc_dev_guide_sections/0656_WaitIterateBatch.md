###### WaitIterateBatch

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

等待 IterateBatch 异步接口或 IterateNBatch 异步接口返回，支持连续输出到 Global Memory。

## 函数原型

```cpp
__aicore__ inline void WaitIterateBatch()
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| 无 | 无 | NA |

## 返回值说明

无

## 约束说明

- 配套 IterateBatch 或 IterateNBatch 异步接口使用。
- 仅支持连续输出至 Global Memory。
- 当使能 MixDualMaster（双主模式）场景时，即模板参数 `enableMixDualMaster` 设置为 `true`，不支持使用该接口。

## 调用示例

```cpp
AscendC::Matmul<aType, bType, cType, biasType> mm;
mm.SetTensorA(gm_a[offsetA]);
mm.SetTensorB(gm_b[offsetB]);
if (tiling.isBias) {
    mm.SetBias(gm_bias[offsetBias]);
}
mm.IterateBatch(gm_c[offsetC], batchA, batchB, false);
// do some others compute
mm.WaitIterateBatch(); // 等待IterateBatch完成
```
