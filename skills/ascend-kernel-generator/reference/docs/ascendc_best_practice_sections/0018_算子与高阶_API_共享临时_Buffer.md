### 算子与高阶 API 共享临时 Buffer

## 优先级
高

## 描述
如果算子使用的高阶 API 需要传入临时 Buffer（如 SoftMax），该临时空间会挤占算子其他计算的空间，从而导致单次计算搬运的数据量变少，搬运的次数变多。此场景可通过共享临时 Buffer 空间，提升单次搬运的数据量，减少搬运的次数，提升内存使用效率。

## 反例
SoftMax 高阶 API 计算需要临时 Buffer 空间，算子在进行其他计算时拥有独立临时 Buffer。UB 空间是固定的，假设可以给 SoftMax 和 Add 能分配临时空间为 64KB，SoftMax 计算需要的临时 Buffer 空间 `tmpSoftmaxBuffer` 占用 32KB，则存储 Add 计算结果的 LocalTensor `tmpSumBuffer` 最多只能分配 32KB。如果 `src0Tensor` 计算的数据量是 512KB，则需要搬运 512 / 32 = 16 次。

```cpp
constexpr int32_t blockLen = 32 * 1024;
TBuf<TPosition::VECCALC> tmpSoftmaxBuf;
pipe.InitBuffer(tmpSoftmaxBuf, softmaxBufSize * sizeof(uint8_t)); // 单独分配Softmax的临时Buf 32KB
TBuf<TPosition::VECCALC> tmpSumBuf;
pipe.InitBuffer(tmpSumBuf, sumBufSize * sizeof(T)); // 单独分配Add的临时Buf，且softmaxBufSize * sizeof(uint8_t) + sumBufSize * sizeof(T) <= 64KB

for (int i = 0; i < 16; i++) {
    LocalTensor<uint8_t> tmpSoftmaxTensor = tmpSoftmaxBuf.Get<uint8_t>(softmaxBufSize);
    SoftMax<T, true, true>(dstTensor, expSumTensor, dstMaxTensor, srcTensor, tmpSoftmaxTensor, tiling);
    
    DataCopy(src0Tensor, src0Gm[i * blockLen / sizeof(T)], Params);
    
    LocalTensor<T> tmpSumTensor = tmpSumBuf.Get<T>(sumBufSize);
    Add<T>(tmpSumTensor, src0Tensor, src1Tensor, count);
}
```

## 正例
SoftMax 高阶 API 计算需要临时 Buffer 空间，算子在进行其他计算时可以共享此临时 Buffer，按照上述假设只需要搬运 512 / 64 = 8 次。

```cpp
constexpr int32_t blockLen = 64 * 1024;
TBuf<TPosition::VECCALC> tmpSharedBuf;
pipe.InitBuffer(tmpSharedBuf, bufferSize); // 共享分配bufferSize = MAX(softmaxBufSize * sizeof(uint8_t), sumBufSize * sizeof(T)) <= 64KB

for (int i = 0; i < 8; i++) {
    LocalTensor<uint8_t> tmpSharedTensor = tmpSharedBuf.Get<uint8_t>(softmaxBufSize);
    SoftMax<T, true, true>(dstTensor, expSumTensor, dstMaxTensor, srcTensor, tmpSharedTensor, tiling);
    
    DataCopy(src0Tensor, src0Gm[i * blockLen / sizeof(T)], Params);
    
    LocalTensor<T> tmpSumTensor = tmpSharedBuf.Get<T>(sumBufSize);
    Add<T>(tmpSumTensor, src0Tensor, src1Tensor, count);
}
```
