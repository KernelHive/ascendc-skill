##### TensorTrait 简介

```markdown
TensorTrait 数据结构是描述 Tensor 相关信息的基础模板类，包含 Tensor 的数据类型、逻辑位置和 Layout 内存布局。借助模板元编程技术，该类在编译时完成计算和代码生成，从而降低运行时开销。

## 头文件

```cpp
#include "kernel_operator_tensor_trait.h"
```

## 原型定义

```cpp
template <typename T, TPosition pos = TPosition::GM, typename LayoutType = Layout<Shape<>, Stride<>>
struct TensorTrait {
    using LiteType = T;
    using LiteLayoutType = LayoutType;
    static constexpr const TPosition tPos = pos; // 该常量成员为后续功能拓展做预留
    
public:
    __aicore__ inline TensorTrait(const LayoutType& t = {});
    __aicore__ inline LayoutType& GetLayout();
    __aicore__ inline const LayoutType& GetLayout() const;
    __aicore__ inline void SetLayout(const LayoutType& t);
};
```

## 模板参数

| 参数名 | 描述 |
|--------|------|
| T | 只支持如下基础数据类型：int4b_t、uint8_t、int8_t、int16_t、uint16_t、bfloat16_t、int32_t、uint32_t、int64_t、uint64_t、float、half。<br>在 TensorTrait 结构体内部，使用 using 关键字定义了一个类型别名 LiteType，与模板参数 T 类型一致。<br>通过 TensorTrait 定义的 LocalTensor/GlobalTensor 不包含 ShapeInfo 信息。<br>例如：LocalTensor<float> 对应的不含 ShapeInfo 信息的 Tensor 为 LocalTensor<TensorTrait<float>>。 |
| pos | 数据存放的逻辑位置，TPosition 类型，默认为 TPosition::GM。 |
| LayoutType | Layout 数据类型，默认为空类型，即 Layout<Shape<>, Stride<>>。<br>输入的数据类型 LayoutType，需满足约束说明。 |

## 成员函数

- `__aicore__ inline TensorTrait(const LayoutType& t = {})`
- `__aicore__ inline LayoutType& GetLayout()`
- `__aicore__ inline const LayoutType& GetLayout() const`
- `__aicore__ inline void SetLayout(const LayoutType& t)`

## 相关接口

```cpp
// TensorTrait 结构构造方法
template <typename T, TPosition pos, typename LayoutType>
__aicore__ inline constexpr auto MakeTensorTrait(const LayoutType& t)

// is_tensorTrait 原型定义
template <typename T> struct is_tensorTrait
```

## 约束说明

- 同一接口不支持同时输入 TensorTrait 类型的 GlobalTensor/LocalTensor 和非 TensorTrait 类型的 GlobalTensor/LocalTensor。
- 非 TensorTrait 类型和 TensorTrait 类型的 GlobalTensor/LocalTensor 相互之间不支持拷贝构造和赋值运算符。
- TensorTrait 特性当前仅支持如下接口：
  - 和 API 配合使用时，当前暂不支持 TensorTrait 结构配置 pos、LayoutType 模板参数，需要使用构造函数构造 TensorTrait，pos、LayoutType 保持默认值即可。
  - DataCopy 切片数据搬运接口需要 ShapeInfo 信息，不支持输入 TensorTrait 类型的 GlobalTensor/LocalTensor。

## TensorTrait 特性支持的接口列表

| 接口分类 | 接口名称 |
|----------|----------|
| 基础API > 内存管理与同步控制 > TQue/TQueBind | AllocTensor、FreeTensor、EnQue、DeQue |
| 基础API > 矢量计算 > 基础算术 | Exp、Ln、Abs、Reciprocal、Sqrt、Rsqrt、Relu、Add、Sub、Mul、Div、Max、Min、Adds、Muls、Maxs、Mins、VectorPadding、BilinearInterpolation、LeakyRelu |
| 基础API > 矢量计算 > 逻辑计算 | And、Or |
| 基础API > 矢量计算 > 复合计算 | AddRelu、AddDeqRelu、SubRelu、MulAddDst、FusedMulAdd、FusedMulAddRelu |
| 基础API > 数据搬运 | DataCopy、Copy、Fixpipe |
| 基础指令 > ISASI（体系结构相关） > 矩阵计算 | InitConstValue、LoadData、LoadDataWithTranspose、SetAippFunctions、LoadImageToLocal、LoadUnzipIndex、LoadDataUnzip、LoadDataWithSparse、Mmad、MmadWithSparse、BroadCastVecToMM、Gemm |
| 基础API > 矢量计算 > 比较指令 | Compare、GetCmpMask、SetCmpMask |
| 基础API > 矢量计算 > 选择指令 | Select、GatherMask |
| 基础API > 矢量计算 > 精度转换指令 | Cast、CastDeq |
| 基础API > 矢量计算 > 归约指令 | ReduceMax、BlockReduceMax、WholeReduceMax、ReduceMin、BlockReduceMin、WholeReduceMin、ReduceSum、BlockReduceSum、WholeReduceSum、RepeatReduceSum、PairReduceSum |
| 基础API > 矢量计算 > 数据转换 | Transpose、TransDataTo5HD |
| 基础API > 矢量计算 > 数据填充 | Brcb |
| 基础API > 矢量计算 > 数据分散/数据收集 | Gather、Gatherb、Scatter |
| 基础API > 矢量计算 > 排序组合（ISASI） | ProposalConcat、ProposalExtract、RpSort16、MrgSort4、Sort32 |
```
