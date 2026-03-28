###### TQueBind 简介

TQueBind 绑定源逻辑位置和目的逻辑位置，根据源位置和目的位置，来确定内存分配的位置、插入对应的同步事件，帮助开发者解决内存分配和管理、同步等问题。TQue 是 TQueBind 的简化模式。通常情况下开发者使用 TQue 进行编程，TQueBind 对外提供一些特殊数据通路的内存管理和同步控制，涉及这些通路时可以直接使用 TQueBind。

如下图的数据通路示意图所示，红色线条和蓝色线条的通路可通过 TQueBind 定义表达，蓝色线条的通路可通过 TQue 进行简化表达。

**表 15-367 TQueBind 和 TQue 对于数据通路的表达**

| 数据通路        | TQueBind 定义                                      | TQue 定义                     |
|-----------------|----------------------------------------------------|-------------------------------|
| GM→VECIN        | `TQueBind<TPosition::GM, TPosition::VECIN, 1>`     | `TQue<TPosition::VECIN, 1>`   |
| VECOUT→GM       | `TQueBind<TPosition::VECOUT, TPosition::GM, 1>`    | `TQue<TPosition::VECOUT, 1>`  |
| VECIN→VECOUT    | `TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>` | -                             |
| GM→A1           | `TQueBind<TPosition::GM, TPosition::A1, 1>`        | `TQue<TPosition::A1, 1>`      |
| GM→B1           | `TQueBind<TPosition::GM, TPosition::B1, 1>`        | `TQue<TPosition::B1, 1>`      |
| GM→C1           | `TQueBind<TPosition::GM, TPosition::C1, 1>`        | `TQue<TPosition::C1, 1>`      |
| A1→A2           | `TQueBind<TPosition::A1, TPosition::A2, 1>`        | `TQue<TPosition::A2, 1>`      |
| B1→B2           | `TQueBind<TPosition::B1, TPosition::B2, 1>`        | `TQue<TPosition::B2, 1>`      |
| C1→C2           | `TQueBind<TPosition::C1, TPosition::C2, 1>`        | `TQue<TPosition::C2, 1>`      |
| CO1→CO2         | `TQueBind<TPosition::CO1, TPosition::CO2, 1>`      | `TQue<TPosition::CO1, 1>`     |
| CO2→GM          | `TQueBind<TPosition::CO2, TPosition::GM, 1>`       | `TQue<TPosition::CO2, 1>`     |
| VECOUT→A1/B1/C1 | `TQueBind<TPosition::VECOUT, TPosition::A1, 1>`<br>`TQueBind<TPosition::VECOUT, TPosition::B1, 1>`<br>`TQueBind<TPosition::VECOUT, TPosition::C1, 1>` | - |
| CO2→VECIN       | `TQueBind<TPosition::CO2, TPosition::VECIN, 1>`    | -                             |

> **说明**：上述表格中的 Cube 相关数据通路建议使用 Cube 高阶 API（如 Matmul）实现，直接使用 TQueBind 控制会相对复杂。

下面通过两个具体的示例展示了矢量编程场景下 TQueBind 的使用方法：

- 如下的编程范式示例，图中的两个队列分别绑定的是 GM→VECIN 和 VECOUT→GM。
- 如果不需要进行 Vector 计算，比如仅需要做格式随路转换等场景，可对上述流程进行优化，对 VECIN 和 VECOUT 进行绑定，绑定的效果可以实现输入输出使用相同 buffer，实现 double buffer。

## 模板参数

```cpp
template <TPosition src, TPosition dst, int32_t depth, auto mask = 0>
class TQueBind {...};
```

**表 15-368 模板参数说明**

| 参数名 | 描述 |
|--------|------|
| src    | 源逻辑位置，支持的 TPosition 可以为 VECIN、VECOUT、A1、A2、B1、B2、CO1、CO2。关于 TPosition 的具体介绍请参考 15.1.4.4.12 TPosition。支持的 src 和 dst 组合请参考表 15-367。 |
| dst    | 目的逻辑位置，TPosition 可以为 VECIN、VECOUT、A1、A2、B1、B2、CO1、CO2。 |
| depth  | TQue 的深度，一般不超过 4。 |
| mask   | 如果用户在某一个 Que 上，数据搬运的时候需要做转换，可以设置为 0 或 1。一般不需要用户配置，默认为 0。<br>设置为 0，代表数据格式从 ND 转换为 NZ，目前仅支持 TPosition 为 A1 或 B1。 |
