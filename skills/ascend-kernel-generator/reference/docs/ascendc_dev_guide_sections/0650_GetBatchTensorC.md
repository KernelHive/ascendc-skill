###### GetBatchTensorC

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

调用一次 `GetBatchTensorC`，会获取 C 矩阵片，该接口可以与 `IterateNBatch` 异步接口配合使用。用于在调用 `IterateNBatch` 迭代计算后，获取一片 `std::max(batchA, batchB) * singleCoreM * singleCoreN` 大小的矩阵分片。

## 函数原型

```cpp
template <bool sync = true>
__aicore__ inline GlobalTensor<DstT> GetBatchTensorC(uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)

template <bool sync = true>
__aicore__ inline void GetBatchTensorC(const LocalTensor<DstT>& c, uint32_t batchA, uint32_t batchB, bool enSequentialWrite = false)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| sync | 当前仅支持异步模式，即该参数只支持取值为 false。 |

### 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| batchA | 输入 | 左矩阵的 batch 数 |
| batchB | 输入 | 右矩阵的 batch 数 |
| enSequentialWrite | 输入 | 该参数预留，开发者无需关注。 |
| c | 输入 | C 矩阵放置于 Local Memory 的地址，用于保存矩阵分片 |

## 返回值说明

`GlobalTensor<DstT>`，返回计算的矩阵分片。

## 约束说明

当使能 MixDualMaster（双主模式）场景时，即模板参数 `enableMixDualMaster` 设置为 true，不支持使用该接口。

## 调用示例

```cpp
// 计算需要多Batch计算循环次数
int for_extent = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / tiling.BatchNum;
mm1.SetTensorA(gm_a[0], isTransposeAIn);
mm1.SetTensorB(gm_b[0], isTransposeBIn);
if (tiling.isBias) {
    mm1.SetBias(gm_bias[0]);
}
// 多batch Matmul计算
mm1.template IterateNBatch<false>(for_extent, batchA, batchB, false);
...other compute
for (int i = 0; i < for_extent; ++i) {
    mm1.template GetBatchTensorC<false>(ubCmatrix);
    ...other compute
}
```
