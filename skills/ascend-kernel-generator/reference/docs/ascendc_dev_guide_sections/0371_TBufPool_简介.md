###### TBufPool 简介

TPipe 可以管理全局内存资源，而 TBufPool 可以手动管理或复用 Unified Buffer/L1 Buffer 物理内存，主要用于多个 stage 计算中 Unified Buffer/L1 Buffer 物理内存不足的场景。

完整样例链接可以参考 [TBufPool 样例](#)。

## 功能图示

下图展示了资源池划分的过程：

1. 通过 `TPipe::InitBuffer` 接口可以申请 Buffer 内存并使用队列进行管理；
2. 通过 `TPipe::InitBufPool` 可以划分出资源池 BufPool1；
3. 通过 `TPipe::InitBufPool` 可以指定 BufPool1 与 BufPool3 地址和长度复用；
4. 通过 `TBufPool::InitBuffer` 及 `TBufPool::InitBufPool` 接口继续将 BufPool1 及 BufPool3 划分成 Buffer 或 TBufPool 资源池。

![图 15-24 BufPool 资源池划分]()

如图示的嵌套关系，最外层 TBufPool（BufPool1 与 BufPool3）需要通过 `TPipe::InitBufPool` 申请并初始化，内层 TBufPool（BufPool2）可以通过 `TBufPool::InitBufPool` 申请并初始化。

## 约束说明

1. TBufPool 必须通过 `TPipe::InitBufPool` 或 `TBufPool::InitBufPool` 接口进行划分和初始化；资源池只能整体划分成部分，无法部分拼接为整体；
2. 不同 TBufPool 资源池切换进行计算时，需要调用 `TBufPool::Reset()` 接口清空已完成计算的 TBufPool，清空后的 TBufPool 资源池及分配的 Buffer 和数据默认无效；
3. 不同资源池间分配的 Buffer 无法混用，避免数据踩踏；
4. `AllocTensor`/`FreeTensor`、`EnQue`/`DeQue` 在切分 TBufPool 资源池时必须成对匹配使用，自动确保同步；
5. 切换资源池的时候，若手写同步，Ascend C 不保证地址读写复用同步，因此不推荐手写同步。
