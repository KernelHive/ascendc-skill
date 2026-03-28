## 如何进行 Tiling 调测

在工程化算子开发过程中，开发者需实现 Tiling 函数，该函数原型是固定的，接受 `TilingContext` 作为输入。框架负责构造 `TilingContext` 并调用 Tiling 函数。若需单独进行 Tiling 调测，开发者可通过 `OpTilingRegistry` 加载编译后的 Tiling 动态库，获取 Tiling 函数的指针并进行调用，调用时 Tiling 函数的 `TilingContext` 入参使用 `ContextBuilder` 构建。

以下是具体步骤：

## 步骤 1：准备 Tiling 动态库

参考工程化算子开发的开发步骤，完成算子实现，并通过算子包编译或算子动态库编译获取对应的 Tiling 动态库文件。

- **算子包编译**：Tiling 实现对应的动态库为算子包部署目录下的 `liboptiling.so`。具体路径可参考“算子包部署”章节。
- **动态库编译**：Tiling 实现集成在算子动态库 `libcust_opapi.so` 中。具体路径可参考“算子动态库和静态库编译”章节。

## 步骤 2：编写测试代码

- 使用 `ContextBuilder` 配置输入输出 Tensor 的形状、数据类型、格式及平台信息等，构建 `TilingContext`。
- 通过 `OpTilingRegistry` 的 `LoadTilingLibrary` 接口加载 Tiling 动态库；使用 `GetTilingFunc` 接口获取 Tiling 函数指针。
- 执行 Tiling 函数，验证其正确性。

```cpp
// test.cpp
#include <iostream>
#include "exe_graph/runtime/storage_shape.h"
#include "tiling/context/context_builder.h"

int main() {
    gert::StorageShape x_shape = {{2, 32}, {2, 32}};
    gert::StorageShape y_shape = {{2, 32}, {2, 32}};
    gert::StorageShape z_shape = {{2, 32}, {2, 32}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());

    auto holder = context_ascendc::ContextBuilder()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .AddInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND, x_shape)
        .AddInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND, y_shape)
        .AddOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND, z_shape)
        .TilingData(param.get())
        .Workspace(ws_size)
        .AddPlatformInfo("Ascendxxxyy")
        .BuildTilingContext();
    auto tilingContext = holder->GetContext<gert::TilingContext>();

    context_ascendc::OpTilingRegistry tmpIns;
    bool flag = tmpIns.LoadTilingLibrary("/your/path/to/so_path/liboptiling.so"); // 加载对应的 Tiling 动态库文件
    if (flag == false) {
        std::cout << "Failed to load tiling so" << std::endl;
        return -1;
    }

    context_ascendc::TilingFunc tilingFunc = tmpIns.GetTilingFunc("AddCustom"); // 获取 AddCustom 算子对应的 Tiling 函数，此处入参为 OpType
    if (tilingFunc != nullptr) {
        ge::graphStatus ret = tilingFunc(tilingContext); // 执行 Tiling 函数
        if (ret != ge::GRAPH_SUCCESS) {
            std::cout << "Exec tiling func failed." << std::endl;
            return -1;
        }
    } else {
        std::cout << "Get tiling func failed." << std::endl;
        return -1;
    }
    return 0;
}
```

## 步骤 3：编译测试代码

```bash
g++ test.cpp -I${INSTALL_DIR}/include -L${INSTALL_DIR}/lib64 -Wl,-rpath,${INSTALL_DIR}/lib64 -ltiling_api -lc_sec -lgraph_base -lregister -lascendalog -lplatform -o test
```

说明：

- `${INSTALL_DIR}` 请替换为 CANN 软件安装后文件存储路径。若安装的是 `Ascend-cann-toolkit` 软件包，以 root 安装举例，则安装后文件存储路径为：`/usr/local/Ascend/ascend-toolkit/latest`。
- 开发者根据需要链接依赖的动态库，必需链接的动态库有：
  - `libtiling_api.so`：Tiling 功能相关的动态库，包含 `ContextBuilder` 类、`OpTilingRegistry` 类等。
  - `libc_sec.so`：安全函数库，`libtiling_api.so` 依赖该库。
  - `libgraph_base.so`：基础数据结构与接口库，`libtiling_api.so` 依赖该库。
  - `libregister.so`：业务函数注册相关库（例如 Tiling 函数注册，算子原型注册等）。
  - `libascendalog.so`：log 库，`libtiling_api.so` 依赖该库。
  - `libplatform.so`：平台信息库，`libtiling_api.so` 依赖该库；Tiling 函数中使用硬件平台信息时，需要依赖该库。

## 步骤 4：执行可执行文件

```bash
./test
```
