### 使能 Iterate 或 IterateAll 异步接口避免 AIC/AIV 同步依赖

【优先级】高

【描述】
在 MIX 场景（即 AIC 和 AIV 混合编程）中，调用 Matmul Iterate 或 IterateAll 时，AIV 会向 AIC 发送消息以启动矩阵乘法计算。

- 若使用 `Iterate<true>` 同步方式（如图 5-28 所示），每次调用都会触发一次消息发送。
- 若使用 `Iterate<false>` 异步方式（如图 5-29 所示），仅第一次调用需要发送消息，后续无需发送，从而减少 Cube 与 Vector 核之间的交互，降低核间通信开销。

因此，在 MIX 场景中推荐使用 `Iterate<false>` 或 `IterateAll<false>` 异步接口。

> 注意：使用异步接口时需要设置 Workspace。

图 5-28 同步方式消息发送示意图

图 5-29 异步方式消息发送示意图

---

## 反例

MIX 场景使用 Iterate 接口的同步方式：

```cpp
TQueBind<TPosition::CO2, TPosition::VECIN> qVecIn;
TQueBind<TPosition::VECIN, TPosition::VECOUT> qVecOut;

mm.SetTensorA(gmA);
mm.SetTensorB(gmB);
int16_t scalar = 2;

while (mm.template Iterate()) {
    auto cInUB = qVecIn.AllocTensor<float>();
    mm.GetTensorC(cInUB);
    qVecIn.EnQue(cInUB);
    cInUB = qVecIn.DeQue<float>();
    auto cOutUB = qVecOut.AllocTensor<float>();
    Muls(cOutUB, cInUB, scalar, baseM * baseN);
    qVecIn.FreeTensor(cInUB);
    // ...
}
```

---

## 正例

MIX 场景使用 Iterate 接口的异步方式：

```cpp
TQueBind<TPosition::CO2, TPosition::VECIN> qVecIn;
TQueBind<TPosition::VECIN, TPosition::VECOUT> qVecOut;

mm.SetTensorA(gmA);
mm.SetTensorB(gmB);
mm.SetWorkspace(workspace, size); // 其中 workspace 为临时空间的物理地址，size 为 singleCoreM * singleCoreN 大小的矩阵 C 占用的内存大小：singleCoreM * singleCoreN * sizeof(float)
int16_t scalar = 2;

while (mm.template Iterate<false>()) {
    auto cInUB = qVecIn.AllocTensor<float>();
    mm.GetTensorC(cInUB);
    qVecIn.EnQue(cInUB);
    cInUB = qVecIn.DeQue<float>();
    auto cOutUB = qVecOut.AllocTensor<float>();
    Muls(cOutUB, cInUB, scalar, baseM * baseN);
    qVecIn.FreeTensor(cInUB);
    // ...
}
```
