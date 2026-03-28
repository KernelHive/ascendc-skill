### Pybind 调用

通过 PyTorch 框架进行模型的训练、推理时，会调用很多算子进行计算，其中的调用方式与 kernel 编译流程有关。对于自定义算子工程，需要使用 PyTorch Ascend Adapter 中的 OP-Plugin 算子插件对功能进行扩展，让 torch 可以直接调用自定义算子包中的算子，详细内容可以参考 10.2 PyTorch 框架；对于 KernelLaunch 开放式算子编程的方式，通过适配 Pybind 调用，可以实现 PyTorch 框架调用算子 kernel 程序。

Pybind 是一个用于将 C++ 代码与 Python 解释器集成的库，实现原理是通过将 C++ 代码编译成动态链接库（DLL）或共享对象（SO）文件，使用 Pybind 提供的 API 将算子核函数与 Python 解释器进行绑定。在 Python 解释器中使用绑定的 C++ 函数、类和变量，从而实现 Python 与 C++ 代码的交互。在 Kernel 直调中使用时，就是将 Pybind 模块与算子核函数进行绑定，将其封装成 Python 模块，从而实现两者交互。

Pybind 调用方式中，使用的主要接口有：

- `c10_npu::getCurrentNPUStream`：获取当前 NPU 流，返回值类型 NPUStream，使用方式请参考《Ascend Extension for PyTorch 自定义 API 参考》中的“（beta）c10_npu::getCurrentNPUStream”章节。
- `ACLRT_LAUNCH_KERNEL`：同 ACLRT_LAUNCH_KERNEL 中 ACLRT_LAUNCH_KERNEL 接口。

算子样例工程请通过如下链接获取：

- 矢量算子样例
- 矢量+矩阵融合算子样例

## 环境准备

基于环境准备，还需要安装以下依赖：

- 安装 PyTorch 框架
- 安装 torch_npu 插件
- 安装 pybind11

```bash
pip3 install pybind11
```

## 工程目录

您可以单击矢量算子样例，获取核函数开发和运行验证的完整样例。样例目录结构如下所示：

```
├── CppExtensions
│   ├── add_custom_test.py  // Python调用脚本
│   ├── add_custom.cpp      // 算子实现
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── pybind11.cpp        // pybind11函数封装
│   └── run.sh              // 编译运行算子的脚本
```

基于该算子工程，开发者进行算子开发的步骤如下：

- 完成算子 kernel 侧实现。
- 编写算子调用应用程序和定义 pybind 模块 pybind11.cpp。
- 编写 Python 调用脚本 add_custom_test.py，包括生成输入数据和真值数据，调用封装的模块以及验证结果。
- 编写 CMake 编译配置文件 CMakeLists.txt。
- 根据实际需要修改编译运行算子的脚本 run.sh 并执行该脚本，完成算子的编译运行和结果验证。

## 算子 kernel 侧实现

请参考 6.2 矢量编程和工程目录中的算子 kernel 实现完成 Ascend C 算子实现文件的编写。

## 算子调用应用程序和定义 pybind 模块

下面代码以 add_custom 算子为例，介绍算子核函数调用的应用程序 pybind11.cpp 如何编写。您在实现自己的应用程序时，需要关注由于算子核函数不同带来的修改，包括算子核函数名，入参出参的不同等，相关 API 的调用方式直接复用即可。

### 步骤 1 按需包含头文件

需要注意的是，需要包含对应的核函数调用接口声明所在的头文件 `aclrtlaunch_{kernel_name}.h`（该头文件为工程框架自动生成），kernel_name 为算子核函数的名称。

```cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "aclrtlaunch_add_custom.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
```

### 步骤 2 应用程序框架编写

需要注意的是，本样例的输入 x，y 的内存是在 Python 调用脚本 add_custom_test.py 中分配的。

```cpp
namespace my_add {
at::Tensor run_add_custom(const at::Tensor &x, const at::Tensor &y) {
}
}
```

### 步骤 3 NPU 侧运行验证

使用 ACLRT_LAUNCH_KERNEL 接口调用算子核函数完成指定的运算。

```cpp
// 运行资源申请，通过c10_npu::getCurrentNPUStream()的函数获取当前NPU上的流
auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
// 分配Device侧输出内存
at::Tensor z = at::empty_like(x);
uint32_t blockDim = 8;
uint32_t totalLength = 1;
for (uint32_t size : x.sizes()) {
    totalLength *= size;
}
// 用ACLRT_LAUNCH_KERNEL接口调用核函数完成指定的运算
ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, acl_stream,
    const_cast<void *>(x.storage().data()),
    const_cast<void *>(y.storage().data()),
    const_cast<void *>(z.storage().data()),
    totalLength);
// 将Device上的运算结果拷贝回Host并释放申请的资源
return z;
```

### 步骤 4 定义 Pybind 模块

将 C++ 函数封装成 Python 函数。PYBIND11_MODULE 是 Pybind11 库中的一个宏，用于定义一个 Python 模块。它接受两个参数，第一个参数是封装后的模块名，第二个参数是一个 Pybind11 模块对象，用于定义模块中的函数、类、常量等。通过调用 `m.def()` 方法，可以将步骤 2 中函数 `my_add::run_add_custom()` 转成 Python 函数 `run_add_custom`，使其可以在 Python 代码中被调用。

```cpp
PYBIND11_MODULE(add_custom, m) { // 模块名add_custom，模块对象m
    m.doc() = "add_custom pybind11 interfaces"; // optional module docstring

    m.def("run_add_custom", &my_add::run_add_custom, ""); // 将函数run_add_custom与Pybind模块进行绑定
}
```

## Python 调用脚本

在 Python 调用脚本中，使用 torch 接口生成随机输入数据并分配内存，通过导入封装的自定义模块 add_custom，调用自定义模块 add_custom 中的 run_add_custom 函数，从而在 NPU 上执行算子。算子核函数 NPU 侧运行验证的步骤如图 1 NPU 侧运行验证原理图。

```python
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os
sys.path.append(os.getcwd())
import add_custom
torch.npu.config.allow_internal_format = False

class TestCustomAdd(TestCase):
    def test_add_custom_ops(self):
        # 分配Host侧输入内存，并进行数据初始化
        length = [8, 2048]

        x = torch.rand(length, device='cpu', dtype=torch.float16)
        y = torch.rand(length, device='cpu', dtype=torch.float16)
        # 分配Device侧输入内存，并将数据从Host上拷贝到Device上
        x_npu = x.npu()
        y_npu = y.npu()
        output = add_custom.run_add_custom(x_npu, y_npu)
        cpuout = torch.add(x, y)
        self.assertRtolEqual(output, cpuout)

if __name__ == "__main__":
    run_tests()
```

## CMake 编译配置文件编写

通常情况下不需要开发者修改编译配置文件，但是了解编译配置文件可以帮助开发者更好的理解编译原理，方便有能力的开发者对 Cmake 进行定制化处理，具体内容请参考 CMake 编译配置文件编写。

## 修改并执行一键式编译运行脚本

您可以基于样例工程中提供的一键式编译运行脚本 run.sh 进行快速编译，在 NPU 侧执行 Ascend C 算子。一键式编译运行脚本主要完成以下功能：

> **须知**
>
> 样例中提供的一键式编译运行脚本并不能适用于所有的算子运行验证场景，使用时请根据实际情况进行修改。
>
> - 根据 Ascend C 算子的算法原理的不同，自行实现输入和真值数据的生成。

完成上述文件的编写后，可以执行一键式编译运行脚本，编译和运行应用程序。

脚本执行方式和脚本参数介绍如下：

```bash
bash run.sh --soc-version=<soc_version>
bash run.sh -v <soc_version>
```

### 表 6-15 脚本参数介绍

| 参数名         | 参数简写 | 参数介绍                                                                 |
|----------------|----------|--------------------------------------------------------------------------|
| `--soc-version` | `-v`     | 算子运行的 AI 处理器型号。                                                |

**说明**

AI 处理器的型号请通过如下方式获取：

- 针对如下产品型号：在安装昇腾 AI 处理器的服务器执行 `npu-smi info` 命令进行查询，获取 Name 信息。实际配置值为 `AscendName`，例如 Name 取值为 `xxxyy`，实际配置值为 `Ascendxxxyy`。
  - Atlas A2 训练系列产品 / Atlas A2 推理系列产品
  - Atlas 200I/500 A2 推理产品
  - Atlas 推理系列产品
  - Atlas 训练系列产品

- 针对如下产品型号，在安装昇腾 AI 处理器的服务器执行 `npu-smi info -t board -i id -c chip_id` 命令进行查询，获取 Chip Name 和 NPU Name 信息，实际配置值为 `Chip Name_NPU Name`。例如 Chip Name 取值为 `Ascendxxx`，NPU Name 取值为 `1234`，实际配置值为 `Ascendxxx_1234`。其中：
  - `id`：设备 id，通过 `npu-smi info -l` 命令查出的 NPU ID 即为设备 id。
  - `chip_id`：芯片 id，通过 `npu-smi info -m` 命令查出的 Chip ID 即为芯片 id。
  - Atlas A3 训练系列产品 / Atlas A3 推理系列产品

该样例支持以下型号：

- Atlas 推理系列产品
- Atlas 训练系列产品
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
