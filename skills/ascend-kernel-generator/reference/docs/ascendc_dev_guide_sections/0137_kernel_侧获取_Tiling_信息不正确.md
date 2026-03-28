## kernel 侧获取 Tiling 信息不正确

## 现象描述

通过算子在 kernel 侧实现代码中添加 PRINTF 打印发现 kernel 侧获取的 Tiling 信息不正确。

例如，增加的打印代码如下：

```c
PRINTF("tiling_data.totalLength: %d tiling_data.tileNum: %d.\n", tiling_data.totalLength, tiling_data.tileNum);
```

打印的 Tiling 数据如下，全为 0：

```
tiling_data.totalLength: 0 tiling_data.tileNum: 0.
```

## 问题根因

kernel 侧获取 Tiling 信息不正确的原因一般有以下两种：

- host 侧计算 Tiling 的逻辑不正确
- kernel 侧核函数的参数未按照正确顺序填写

## 处理步骤

### 步骤 1

参考如下示例，打印 TilingData 的数据，确认 host 侧序列化保存的 TilingData 是否正确。如果此时打印值有误，说明 Tiling 的计算逻辑可能不正确，需要进一步检查 host 侧 Tiling 实现代码，排查计算逻辑是否有误。

```cpp
std::cout << *reinterpret_cast<uint32_t *>(context->GetRawTilingData()->GetData()) << std::endl; // 按照实际数据类型打印 TilingData 第一个参数值，如需确认其他值，取值指针向后偏移即可
```

### 步骤 2

如果上一步骤中打印的 TilingData 正确，需要排查 kernel 侧核函数的参数是否按照正确顺序填写。

使用 msOpGen 工具创建算子工程，并基于工程进行 kernel 侧算子开发时，核函数的定义模板已通过 msOpGen 工具自动生成，样例如下所示。参数按照“输入、输出、workspace、tiling”的顺序排布。请检查是否调整过参数顺序导致和正确顺序不一致。

```cpp
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling); // 获取 Tiling 参数
    // TODO: user kernel impl
}
```

---

## 现象描述

使用工程化算子开发方式，基于自定义算子工程进行算子开发。编译算子时失败，报如下错误：

```
[ERROR] [ascendxxxx] PowerCustom_88a695f03edfbc0af76b9eaae9e4556c error: out of jump/jumpc imm range
```

## 问题根因

该编译错误的原因是算子 kernel 代码过大，导致在编译时跳转指令跳转的偏移值超过了限定的大小（int16_t 的数据范围），可通过添加编译选项 `-mllvm -cce-aicore-jump-expand=true` 通过间接跳转的方式来避免该问题，让编译器能够正常编译。

## 处理步骤

### 步骤 1

在 kernel 侧的 CMakeLists 中通过 `add_ops_compile_options` 针对报错算子添加编译选项 `-mllvm -cce-aicore-jump-expand=true`，示例如下：

```cmake
add_ops_compile_options(PowerCustom OPTIONS -mllvm -cce-aicore-jump-expand=true)
```

`add_ops_compile_options` 的具体使用方法请参考“支持自定义编译选项”。

### 步骤 2

重新编译该算子。正常编译无报错。

---

## 现象描述

1. 基于 CANN-7.2 及之前版本（<=7.2）的 CANN 开发套件包，编译含有 Matmul 高阶 API 的自定义算子包，将编译后的自定义算子包安装至 CANN-7.3 及之后版本（>=7.3）的 CANN 包环境，然后对该含有 Matmul 高阶 API 的算子，执行图模式在线编译时，报如下错误：

```
res = struct.unpack_from(fmt_str, tiling_data, offset + unpack_size)
struct.error: unpack_from requires a buffer of at least 52 bytes for unpacking 4 bytes at offset 48
```

2. 基于 CANN-7.2 及之前版本（<=7.2）的 CANN 开发套件包，编译 sample 样例仓中含有 Matmul 高阶 API 的算子，例如 MatmulLeakyReluCustomSample，将编译后的自定义算子包安装至 CANN-7.3 及之后版本（>=7.3）的 CANN 包环境，然后对该含有 Matmul 高阶 API 的算子，执行单算子 API 的调用时，报如下错误：

```
ERROR：acl executable run failed! please check your project!
```

## 问题根因

该错误的原因是编译自定义算子包的软件版本过老，可通过更新自定义算子包编译环境上的 CANN 开发套件包版本，然后重新编译和部署自定义算子包，来避免出现该问题。

## 处理步骤

### 步骤 1

查看自定义算子包编译时使用的 CANN 开发套件包版本号，示例如下：

```bash
cd ${CANN 包安装路径}
cat version.cfg
```

输出示例：

```
# version: 1.0
runtime_running_version=[7.2.T11.0.B218:8.0.RC2.alpha001]
runtime_upgrade_version=[7.2.T11.0.B218:8.0.RC2.alpha001]
runtime_installed_version=[7.2.T11.0.B218:8.0.RC2.alpha001]
```

### 步骤 2

基于 CANN-7.3 及之后版本（>=7.3）的 CANN 开发套件包，重新编译该自定义算子包。部署编译生成的自定义算子包后，正常编译或者执行算子，无报错。重新编译和部署自定义算子包的具体方法可参考“6.7.6 算子包编译”。
