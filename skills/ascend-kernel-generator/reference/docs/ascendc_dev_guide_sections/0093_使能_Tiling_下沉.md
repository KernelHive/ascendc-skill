## 使能 Tiling 下沉

在静态图模式下，可以通过整图下沉优化调度性能。将完整的计算图一次性下发至 Device 侧，后续执行则无需 Host 参与，由 Device 自主完成计算，从而减少 Host-Device 交互开销，提升执行效率。部分算子的 Tiling 计算依赖运行时输入的具体数值（Tiling 值依赖），需在执行时动态计算 Tiling 参数。针对该场景，可采用 Tiling 下沉优化方案：将 Tiling 计算下沉至 Device 侧的 AI CPU 上执行，从而实现计算全程在 Device 侧高效完成。

## 说明

- 基于新版本 CANN 包（支持 Tiling 下沉特性）编译生成的 Tiling 下沉算子，不兼容旧版 CANN（不支持 Tiling 下沉特性）运行环境。
- 当前仅融合算子（矢量计算和矩阵计算融合）支持进行 Tiling 下沉。
- Tiling 下沉功能仅支持如下产品型号：
  - Atlas A3 训练系列产品 / Atlas A3 推理系列产品
  - Atlas A2 训练系列产品 / Atlas A2 推理系列产品

## 自定义算子使能 Tiling 下沉步骤

自定义算子使能 Tiling 下沉的步骤如下，完整样例请参考 Tiling 下沉算子样例。

Tiling 下沉场景下，算子工程的 `op_host` 目录结构如下，Tiling 实现文件需单独放在一个 cpp 文件中，示例中为 `add_custom_tiling_sink_tiling.cpp`。

```
├── op_host
│   ├── add_custom_tiling_sink.cpp          // 算子原型定义、InferShape、InferDataType 实现
│   ├── add_custom_tiling_sink_tiling.cpp   // Tiling 函数实现
│   ├── add_custom_tiling_sink_tiling.h     // TilingData 结构体定义、Tiling 函数声明
│   └── CMakeLists.txt
```

以 AddCustom 算子为例，讲解关键代码文件的具体实现方法如下：

### TilingData 结构体定义、Tiling 函数声明头文件

头文件 `add_custom_tiling_sink_tiling.h` 中：

- 进行 TilingData 结构体的定义
- 进行 Tiling 实现函数的声明

```cpp
#ifndef ADD_CUSTOM_TILING_SINK_TILING_H
#define ADD_CUSTOM_TILING_SINK_TILING_H
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingSinkTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustomTilingSink, TilingSinkTilingData) // Tiling 结构体定义

ge::graphStatus AddCustomSinkTilingFunc(gert::TilingContext* context); // Tiling 函数声明
} // namespace optiling
#endif // ADD_CUSTOM_TILING_SINK_TILING_H
```

### 算子原型定义、InferShape、InferDataType 实现文件

文件 `add_custom_tiling_sink.cpp` 需包含 `add_custom_tiling_sink_tiling.h`，进行 Tiling 函数和算子原型定义的关联。

Tiling 下沉仅适用于存在 Tiling 值依赖（即当 InferShape 不依赖输入值，仅 Tiling 计算需要输入值）且算子输入为非 Const 类型的场景，本示例中的输入 y 通过 `ValueDepend` 配置了非 Const 类型的 Tiling 值依赖。

```cpp
#include "add_custom_tiling_sink_tiling.h" // 包含头文件

// ...

namespace ops {
class AddCustomTilingSink : public OpDef {
public:
explicit AddCustomTilingSink(const char *name) : OpDef(name)
{
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    this->Input("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .ValueDepend(OPTIONAL, DependScope::TILING); // 表示输入 y 为 Tiling 值依赖
    this->Output("z")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    this->AICore().SetTiling(optiling::AddCustomSinkTilingFunc); // Tiling 函数和算子原型定义的关联

    // 请替换为实际的昇腾 AI 处理器型号
    this->AICore().AddConfig("ascendxxx");
}
};
OP_ADD(AddCustomTilingSink);
} // namespace ops
```

### Tiling 函数的实现文件

文件 `add_custom_tiling_sink_tiling.cpp`：

- Tiling 函数中通过判断值依赖 InputTensor 即输入 y 的数据指针是否为空指针来确认当前是否处于编译期。Tiling 下沉场景，编译期需要为算子分配内存，包括其所需的 workspace。为了保证运行时的高效性，编译期应根据算子的执行需求，合理设置所需的 workspace 最大值，以避免内存不足或浪费。AddCustomTilingSink 样例不需要用户 workspace，不涉及设置，此处设置为固定值仅作为示例。
- 完成下沉 Tiling 函数注册：包含 `device_op_impl_registry.h` 头文件，使用宏 `DEVICE_IMPL_OP_OPTILING` 进行注册。

```cpp
#include "add_custom_tiling_sink_tiling.h"
#include "register/device_op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static constexpr uint32_t BLOCK_DIM = 8;
static constexpr uint32_t TILE_NUM = 3;
static constexpr size_t MAX_WORKSPACE_SIZE = 32; // 算子所需用户 workspace 空间最大值，AddCustomTilingSink 算子本身逻辑无需用户 workspace 空间，此处设置为固定值仅作为示例
static constexpr size_t DEFAULT_WORKSPACE_SIZE = 0;
ge::graphStatus AddCustomSinkTilingFunc(gert::TilingContext *context)
{
    TilingSinkTilingData tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();

    currentWorkspace[0] = sysWorkspaceSize + DEFAULT_WORKSPACE_SIZE; // 设置运行时 workspace 大小，此处为系统 workspace 空间 + 用户 workspace 空间
    if (context->GetInputTensor(1) != nullptr && context->GetInputTensor(1)->GetData<float>() == nullptr) {
        // 通过判断值依赖 InputTensor 的 Data 是否为空指针来确认当前是否处于编译期。
        // Tiling 下沉场景，编译期需要为算子分配内存，包括其所需的 workspace。为了保证运行时的高效性，编译期应根据算子的执行需求，合理设置所需的 workspace 最大值，以避免内存不足或浪费。
        currentWorkspace[0] = sysWorkspaceSize + MAX_WORKSPACE_SIZE; // 设置编译期 workspace 大小，此处为系统 workspace 空间 + 用户 workspace 空间最大值
    }
    return ge::GRAPH_SUCCESS;
}
DEVICE_IMPL_OP_OPTILING(AddCustomTilingSink).Tiling(optiling::AddCustomSinkTilingFunc); // 下沉 Tiling 函数注册
} // namespace optiling
```

### 算子核函数实现

当前 Tiling 下沉仅支持融合算子，为了模拟融合算子场景，通过 `KERNEL_TASK_TYPE_DEFAULT` 接口强制指定算子在 AIC、AIV 混合场景运行。

```cpp
extern "C" __global__ __aicore__ void add_custom_tiling_sink(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // 将算子强制指定在 AIC、AIV 混合场景运行，模拟融合算子场景
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::KernelAdd op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
```

### 修改编译脚本

修改 `op_host` 目录下的编译脚本 `CMakeLists.txt`，添加 Tiling 下沉编译命令。具体代码如下所示：

```cmake
# 通过 ascendc_device_library 添加 Tiling 下沉编译任务
ascendc_device_library(
    TARGET cust_opmaster # 任务名称，固定为 cust_opmaster
    OPTION SHARED        # 动态库（当前仅支持动态库入图下沉）
    SRC ${CMAKE_CURRENT_SOURCE_DIR}/add_custom_tiling_sink_tiling.cpp # Tiling 函数实现代码源文件
)
```
