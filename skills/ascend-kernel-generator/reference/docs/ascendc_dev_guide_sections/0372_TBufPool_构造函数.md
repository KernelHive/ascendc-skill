###### TBufPool 构造函数

## 功能说明
创建 TBufPool 对象时，初始化数据成员。

## 函数原型
```cpp
template <TPosition pos, uint32_t bufIDSize = 4>
__aicore__ inline TBufPool();
```

## 参数说明

### 模板参数说明

| 参数名 | 说明 |
|--------|------|
| pos | TBufPool 逻辑位置，可以为 VECIN、VECOUT、VECCALC、A1、B1、C1。关于 TPosition 的具体介绍请参考 15.1.4.4.12 TPosition。 |
| bufIDSize | TBufPool 可分配 Buffer 数量，默认为 4，不超过 16。<br>对于非共享模式的资源分配，在本 TBufPool 上再次申请 TBufPool 时，申请的 bufIDSize 不能超过原 TBufPool 剩余可用的 Buffer 数量。<br>对于共享模式的资源分配，在本 TBufPool 上再次申请 TBufPool 时，申请的 bufIDSize 不能超过原 TBufPool 设置的 Buffer 数量。 |

## 约束说明
无。
