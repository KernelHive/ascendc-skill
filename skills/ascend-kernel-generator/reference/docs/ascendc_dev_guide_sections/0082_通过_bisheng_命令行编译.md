## 通过 bisheng 命令行编译

毕昇编译器是一款专为昇腾AI处理器设计的编译器，支持异构编程扩展，可以将用户编写的昇腾算子代码编译成二进制可执行文件和动态库等形式。毕昇编译器的可执行程序命名为 `bisheng`，支持 x86、aarch64 等主机系统，并且原生支持设备侧 AI Core 架构指令集编译。通过使用毕昇编译器，用户可以更加高效地进行针对昇腾 AI 处理器的编程和开发工作。

## 入门示例

以下是一个使用毕昇编译器编译静态 Shape 的 `add_custom` 算子入门示例。该示例展示了如何编写源文件 `add_custom.cpp` 以及具体的编译命令。通过这个示例，您可以了解如何使用毕昇编译器进行算子编译。

### 步骤1 包含头文件

在编写算子源文件时，需要包含必要的头文件。

```cpp
// 头文件
#include "data_utils.h"
#include "acl/acl.h"
#include "kernel_operator.h"
```

### 步骤2 核函数实现

- 核函数支持模板。
- 核函数入参支持传入用户自定义的结构体，比如示例中用户自定义的 `AddCustomTilingData` 结构体。

```cpp
// 用户自定义的TilingData结构体
struct AddCustomTilingData {
    uint32_t totalLength;
    uint32_t tileNum;
};

// Kernel核心实现逻辑，包括搬运，计算等
template <typename T>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    // ...
};

// 核函数
// 核函数支持模板，核函数入参支持传入用户自定义的结构体
template <typename T>
__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 该算子执行时仅启动AI Core上的Vector核
    KernelAdd<T> op;
    op.Init(x, y, z, tiling.totalLength, tiling.tileNum);
    op.Process();
}
```

### 步骤3 Host侧调用函数逻辑

包括内存申请和释放，初始化和去初始化，内核调用符调用核函数等。

```cpp
int32_t main(int32_t argc, char *argv[])
{
    gen_data(); // 生成输入输出数据
    uint32_t blockDim = 8;
    size_t inputByteSize = 8 * 1024 * sizeof(uint32_t);
    size_t outputByteSize = 8 * 1024 * sizeof(uint32_t);
    AddCustomTilingData tiling;
    tiling.totalLength = 8192;
    tiling.tileNum = 8;

    // 初始化
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    // 分配Host内存
    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    // 分配Device内存
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // Host内存初始化

    ReadFile("input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("input_y.bin", inputByteSize, yHost, inputByteSize);
    // 将数据从Host上拷贝到Device上
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    // 用内核调用符<<<...>>>调用核函数完成指定的运算
    add_custom<float><<<blockDim, nullptr, stream>>>(xDevice, yDevice, zDevice, tiling);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    // 将Device上的运算结果拷贝回Host
    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("output_z.bin", zHost, outputByteSize);
    // 释放申请的资源
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    // 去初始化
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    // 验证结果
    std::string output_file = "output_z.bin";
    std::string golden_file = "golden.bin";
    if (verify_result(output_file, golden_file)) {
        std::cout << "验证通过！" << std::endl;
        return 0;
    } else {
        std::cout << "验证失败！" << std::endl;
        return 1;
    }
    return 0;
}
```

### 步骤4 采用如下的编译命令进行编译

- `-x asc`：文件名为 cpp 时，指定输入文件的语言为 Ascend C 编程语言。
- `-o add_custom`：指定输出文件名为 `add_custom`。
- `--npu-arch dav-2201`：指定 NPU 的架构版本为 `dav-2201`。`dav-` 后为 NPU 架构版本号，各产品型号对应的架构版本号请通过表 13-1 进行查询。

```bash
bisheng -x asc add_custom.cpp -o add_custom --npu-arch dav-2201
```

### 步骤5 执行可执行文件

```bash
./add_custom
```

## 程序的编译与执行

通过毕昇编译器可以将算子源文件编译为当前平台的可执行文件或算子动态库、静态库。

### 编译生成可执行文件

```bash
# 1.编译hello_world.cpp为当前平台可执行文件
# bisheng [算子源文件] -o [输出产物名称] --npu-arch [NPU架构版本号]，常见参数顺序与g++保持一致。
bisheng -x asc hello_world.cpp -o hello --npu-arch dav-xxxx
```

生成的可执行文件可通过如下方式执行：

```bash
./hello
```

### 编译生成算子动态库

```bash
# 2.编译add_custom_base.cpp生成算子动态库
# bisheng -shared [算子源文件] -o [输出产物名称] --npu-arch [NPU架构版本号]
# 动态库
bisheng -shared -x asc add_custom_base.cpp -o libadd.so --npu-arch dav-xxxx
```

### 编译生成算子静态库

```bash
# 3.编译add_custom_base.cpp生成算子静态库
bisheng -lib [算子源文件] -o [输出产物名称] --npu-arch [NPU架构版本号]
# 静态库
bisheng -lib -x asc add_custom_base.cpp -o libadd.a --npu-arch dav-xxxx
```

在命令行编译场景下，可以按需链接需要的库文件，常见的库文件请参考常用的链接库。编译时会默认链接表 7-3 中列出的库文件。注意如下例外场景：在使用 `g++` 链接 asc 代码编译生成的静态库时，需要手动链接默认链接库。
