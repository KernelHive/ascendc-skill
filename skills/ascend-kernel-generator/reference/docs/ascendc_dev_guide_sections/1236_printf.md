#### printf

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

该接口提供CPU域/NPU域调试场景下的格式化输出功能。

在算子kernel侧实现代码中需要输出日志信息的地方调用printf接口打印相关内容。样例如下：

```cpp
#include "kernel_operator.h"
AscendC::printf("fmt string %d\n", 0x123);
AscendC::PRINTF("fmt string %d\n", 0x123);
```

## 注意

printf（PRINTF）接口打印功能会对算子实际运行的性能带来一定影响，通常在调测阶段使用。开发者可以按需通过如下方式关闭打印功能。

### 自定义算子工程

修改算子工程op_kernel目录下的CMakeLists.txt文件，首行增加编译选项`-DASCENDC_DUMP=0`，关闭ASCENDC_DUMP开关，示例如下：

```cmake
// 关闭所有算子的打印功能
add_ops_compile_options(ALL OPTIONS -DASCENDC_DUMP=0)
```

### Kernel直调工程

- 修改cmake目录下的npu_lib.cmake文件，在ascendc_compile_definitions命令中增加`ASCENDC_DUMP=0`宏定义来关闭NPU侧ASCENDC_DUMP开关。

示例如下：

```cmake
// 关闭所有算子的打印功能
ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
ASCENDC_DUMP=0
)
```

- 修改cmake目录下的cpu_lib.cmake文件，在target_compile_definitions命令中增加`ASCENDC_DUMP=0`宏定义来关闭CPU侧ASCENDC_DUMP开关。

示例如下：

```cmake
target_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
ASCENDC_DUMP=0
)
```

需要注意的是，关闭CPU侧的打印开关时，只对PRINTF接口生效，对printf不生效。

## 打印结果说明

NPU模式下，printf打印结果的最前面会自动打印CANN_VERSION_STR值与CANN_TIMESTAMP值。其中，CANN_VERSION_STR与CANN_TIMESTAMP为宏定义：

- CANN_VERSION_STR代表CANN软件包的版本号信息，形式为字符串
- CANN_TIMESTAMP为CANN软件包发布时的时间戳，形式为数值(uint64_t)

开发者也可在代码中直接使用这两个宏。printf打印结果示例如下：

```
CANN Version: XXX.XX, TimeStamp: 20240807140556417
fmt string 291
fmt string 291
```

## 输出方式

根据算子执行方式的不同，printf的打印结果输出方式不同：

- 动态图或者单算子直调场景下，待输出内容会被解析并打印在屏幕上
- 静态图场景下，整图算子需要全下沉到NPU侧执行，无法直接调用接口打印出单个算子的信息，因此需要在模型执行完毕后，将待输出内容落盘在dump文件中，dump文件需要通过工具解析为可读内容

### dump文件落盘路径

按照优先级排列如下：

- 如果开启了Data Dump功能，dump文件落盘到开发者配置的dump_path路径下。如何开启Dump功能依赖于具体的网络运行方式。以TensorFlow在线推理为例，通过enable_dump、dump_path、dump_mode等参数进行配置。配置方式可参考《TensorFlow 2.6.5模型迁移指南》中的API参考 > TF Adapter 接口（2.x）> npu.global_options > 配置参数说明章节。
- 如果未开启Data Dump功能，但配置了ASCEND_WORK_PATH环境变量，dump文件落盘到ASCEND_WORK_PATH下的printf目录下。ASCEND_WORK_PATH环境变量的配置方式可参考《环境变量参考》。
- 如果未开启Data Dump功能也没有配置ASCEND_WORK_PATH环境变量，dump文件落盘到当前程序执行目录下的printf路径下。

### 解析dump文件

使用show_kernel_debug_data工具将dump二进制文件解析为用户可读内容，命令格式如下：

```bash
show_kernel_debug_data bin_file output_dir
```

show_kernel_debug_data的具体使用方法请参考13.3 show_kernel_debug_data工具。

## 函数原型

```cpp
void printf(__gm__ const char* fmt, Args&&... args)
void PRINTF(__gm__ const char* fmt, Args&&... args)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| fmt | 输入 | 格式控制字符串，包含两种类型的对象：普通字符和转换说明。<br>● 普通字符将原样不动地打印输出。<br>● 转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型。<br>支持的转换类型包括：<br>- %d / %i：输出十进制数，支持打印的数据类型：bool/int8_t/int16_t/int32_t/int64_t<br>- %f：输出实数，支持打印的数据类型：float/half/bfloat16_t<br>- %x：输出十六进制整数，支持打印的数据类型：int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t<br>- %s：输出字符串<br>- %u：输出unsigned类型数据，支持打印的数据类型：bool/uint8_t/uint16_t/uint32_t/uint64_t<br>- %p：输出指针地址<br><br>注意：<br>● 上文列出的数据类型是NPU域调试支持的数据类型，CPU域调试时，支持的数据类型和C/C++规范保持一致。<br>● 在转换类型为%x，即输出十六进制整数时，NPU域上的输出为64位，CPU域上的输出为32位。 |
| args | 输入 | 附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。 |

## 返回值说明

无

## 约束说明

- 本接口不支持打印除换行符之外的其他转义字符。
- 如果开发者需要包含标准库头文件stdio.h和cstdio，请在kernel_operator.h头文件之前包含，避免printf符号冲突。
- 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印。
- 算子入图场景，若一个动态Shape模型中有可下沉的部分，框架内部会将模型拆分为动态调度和下沉调度（静态子图）两部分，静态子图中的算子不支持该printf特性。

## 调用示例

```cpp
#include "kernel_operator.h"

// 整型打印：
AscendC::printf("fmt string %d\n", 0x123);
AscendC::PRINTF("fmt string %d\n", 0x123);

// 浮点型打印：
float a = 3.14;
AscendC::printf("fmt string %f\n", a);
AscendC::PRINTF("fmt string %f\n", a);

// 指针打印：
int *b;
AscendC::printf("TEST %p\n", b);
AscendC::PRINTF("TEST %p\n", b);
```

NPU模式下，程序运行时打印效果如下：

```
CANN Version: XXX.XX, TimeStamp: 20240807140556417
fmt string 291
fmt string 291
fmt string 3.140000
fmt string 3.140000
TEST 0x12c08001a000
TEST 0x12c08001a000
```
