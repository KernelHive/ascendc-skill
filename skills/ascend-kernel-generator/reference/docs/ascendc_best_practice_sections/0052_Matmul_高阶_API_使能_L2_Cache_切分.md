### Matmul 高阶 API 使能 L2 Cache 切分

## 案例介绍

本案例呈现了在 Matmul 计算过程中，输入和输出的数据总量超过 L2 Cache 大小时，通过 L2 Cache 数据切分对算子性能的提升效果。使能 L2 Cache 切分的完整样例请参考 L2 Cache 切分的算子样例。

本案例使用的 AI 处理器的 L2 Cache 大小为 192MB，L2 Cache 纯读带宽约为 GM 的 3 到 4 倍，两者之间存在较大差距。在搬入或搬出相同数据量的情况下，访问 L2 Cache 内的数据比访问 GM 更快。若数据无法命中 L2 Cache，即需要访问的数据不在 L2 Cache 内，导致需要去 GM 上读写，带宽利用效率较低，最终算子搬入或搬出数据成为算子整个运行过程的性能瓶颈。

## 使能 L2 Cache 切分的适用场景

- 输入和输出的数据量超过 L2 Cache 的大小。

本案例的算子规格如下：

**表 6-7 算子规格**

| 输入 | Shape       | Data type | Format |
|------|-------------|-----------|--------|
| a    | 30720, 1024 | float16   | ND     |
| b    | 4096, 1024  | float16   | ND     |

## 获取性能数据

使用 msProf 工具获取算子仿真流水图和上板 Profiling 数据。因为 L2 Cache 切分功能主要利用带宽更大的 L2 Cache，减少 MTE2 数据搬运开销，所以重点分析 MTE2 的流水。

## 分析主要瓶颈点

当前案例基于 Tiling 全量常量化进一步优化，Tiling 全量常量化请参考 6.4.6 Matmul 高阶 API 使能 Tiling 全量常量化案例。优化前的 Profiling 数据如下，C 列的 aic_time 是 867us，K 列的 aic_mte2_time 是 861.9us，MTE2 占比为 99%，MTE2 数据搬运是当前算子性能的瓶颈。

## 设计优化方案

### 优化点一：调整切块大小和计算次数

- 优化前，输入数据不进行切分，所有核一次计算全部数据。如下图所示，图中数字表示核 id，24 个核一次计算 A 和 B 矩阵的所有数据。
- 优化后，输入数据被切分多次，所有核分多次计算，每个核单次计算只依赖切分后的数据量。L2 Cache 切分方案确保单次计算的数据都在 L2 Cache 缓存中，搬运输入数据的效率更高。

**图 6-33 优化点一示意图**

### 优化点二：选择拖尾较小的 L2 Cache 切分方案

结合 5.7.2 核间负载均衡的原理，AI 处理器的物理核数固定，当数据进行 L2 Cache 切分之后，可能出现部分核有计算拖尾的情况，即每次所有核总计算量除以每个核单次处理的数据量不能被核数整除，导致每次计算的最后需要部分尾核计算剩余数据。而在尾核计算时，部分核始终处于空闲状态，导致算子的整体性能变差。

下图中标黄的数据块就是尾块数据，左边方案由于拖尾，每次计算中 0、1、2、3 核多执行一次处理剩余数据。为达到全局负载最优，调整拖尾核的位置，如右边方案所示，完成所有计算时，0 到 7 核均多一次数据块的计算。

在实际场景中，满足切分后的数据量小于 L2 Cache 大小的前提下，拖尾越小越好。基于这个原则可以确定 L2 Cache 切分块数。

**图 6-34 优化点二示意图**

### 优化点三：错位分核，减少左右矩阵同地址冲突问题

- **同地址冲突**：多核并发执行 Matmul 计算时，如果多核同时访问输入矩阵的相同地址，会导致地址冲突，影响性能。
- 在 M 和 N 方向，将矩阵数据 L2 Cache 切分为大数据块，然后在数据块间错位分核，即将每个数据块依次沿对角线分配给不同的核处理，有效减少同地址冲突的问题。比如，在处理同一行的尾块数据 0，1，2，3 时，如果顺序分配执行的核，多核会同时读同一行左矩阵数据，导致读读冲突。若按照对角线方式分配执行的核，在对角线上的尾块数据被分配给核 0，1，2，3 计算，多核访问不同行的左矩阵数据，将减少同地址冲突的次数。

**图 6-35 优化点三示意图**

Matmul API 使能 L2 Cache 切分的完整样例请参考 L2 Cache 切分的算子样例。实现 L2 Cache 切分的关键步骤如下：

### 步骤 1：判断是否需要进行 L2 Cache 切分

如果数据总量超过设定的 L2 Cache 大小，则计算 L2 Cache 切分数目。

```cpp
bool smallDim = mTileNum_ < L1_MIN_UST_DIM && nTileNum_ < L1_MIN_UST_DIM;
if (smallDim || (!EnableL2Tile())) { // 判断计算数据总量是否小于L2Cache阈值
    mL2TileNum_ = mTileNum_;
    nL2TileNum_ = nTileNum_;
    mL2BlockNum_ = 1;
    nL2BlockNum_ = 1;
    return; // 不需要切分，提前返回
}
InitL2TileTail(); // 计算L2切分
```

### 步骤 2：基于负载均衡原则，计算 L2 Cache 切分的份数

m 方向 L2 Cache 切分数：`mL2TileNum_`，n 方向 L2 Cache 切分数：`nL2TileNum_`。

```cpp
int64_t mConflict = INT64_MAX;
int64_t nConflict = INT64_MAX;
constexpr bool isNMajor = l1N > l1M; // 根据shape大小，判断主维度
for (int64_t i = maxMajor; i >= L1_MIN_UST_DIM; i--) {
    for (int64_t j = maxMinor; j >= minMinor; j--) {
        if (GetTotalSize(j * l1M, i * l1N, k_) <= L2_TILE_THRESHOLD) { // 确保分块小于L2Cache阈值
            uint64_t mConflictTmp = AscendC::Ceil(blockNum_, mL2TileNumTailTmp); // 计算负载冲突值
            uint64_t nConflictTmp = AscendC::Ceil(blockNum_, nL2TileNumTailTmp);
            if (mConflict >= mConflictTmp && nConflict >= nConflictTmp) { // 若冲突值更小，更新分块数量
                mConflict = mConflictTmp;
                nConflict = nConflictTmp;
                mL2TileNum_ = curMajorDim;
                nL2TileNum_ = curMinorDim;
            }
        }
    }
}
```

### 步骤 3：错位分核

输入当前数据块的下标，获取按对角线分配的核的下标。

```cpp
__aicore__ inline BlockCoord GetBlockCoord(int64_t tileIdx) {
    GetCommonTileIndex(tileIdx);
    int64_t mTileIdx = newBlockIdx_ % mL2TileNumTmp_;
    mTileIdx = mTileIdx + mL2Idx_ * mL2TileNum_;
    int64_t nTileIdx = 0;
    if (mL2TileNumTmp_ != 0 && nL2TileNumTmp_ != 0) {
        int64_t tmp = newBlockIdx_ / CalcLcm(mL2TileNumTmp_, nL2TileNumTmp_);
        nTileIdx = (newBlockIdx_ + tmp) % nL2TileNumTmp_;
    }
    nTileIdx = nTileIdx + nL2Idx_ * nL2TileNum_;
    return {mTileIdx * l1M, nTileIdx * l1N, 0};
}
```

### 步骤 4：设置左右矩阵，循环多次计算 Matmul

根据前序步骤计算的 L2 Cache 切分数和执行核的下标，循环多次计算 Matmul。

```cpp
L2CacheOpt l2Opt(shapes, blockNum);
matmulObj.SetOrgShape(shapes.m, shapes.n, shapes.k);
for (int64_t tileIdx = curBlockIdx; tileIdx < l2Opt.GetTileNum(); tileIdx += blockNum) {
    auto blockShape = l2Opt.GetBlockShape(tileIdx); // 获取单次计算L2切分块大小
    if (Get<0>(blockShape) <= 0 || Get<1>(blockShape) <= 0) {
        return;
    }
    auto blockCoord = l2Opt.GetBlockCoord(tileIdx);
    // 获取当前执行计算的核的下标blockCoord
    matmulObj.SetTail(Get<0>(blockShape), Get<1>(blockShape), Get<2>(blockShape));
    const auto& offsetCoord = CalcOffset(shapes, blockCoord); // 基于下标计算矩阵偏移
    int64_t offsetA = Get<0>(offsetCoord);
    int64_t offsetB = Get<1>(offsetCoord);
    int64_t offsetC = Get<2>(offsetCoord);

    matmulObj.SetTensorA(aGlobal[offsetA], false);
    matmulObj.SetTensorB(bGlobal[offsetB], false);
    if (shapes.isBias) {
        matmulObj.SetBias(biasGlobal);
    }
    matmulObj.IterateAll(cGlobal[offsetC]); // 计算L2切分块
}
matmulObj.End();
```

## 验证优化方案性能收益

优化后的 Profiling 数据如下，C 列的 aic_time 为 805.6us，相比于优化前，总执行时间降低了约 7.1%，MTE2 搬运时间降低了约 10.7%。

## 总结

在 Matmul 计算数据量超过 L2 Cache 大小的场景下，可以考虑使能 L2 Cache 切分，提高 L2 Cache 命中率，利用 L2 Cache 高带宽特性提升算子性能。
