# cann-ops 昇腾基础算子仓库快速上手指南

## 1. 项目介绍

cann-ops是基于昇腾硬件的基础算子仓库，欢迎开发者学习、使用和贡献基于昇腾平台的算子代码。

## 2. 仓库结构

本仓库主要包含使用[AscendC算子编程语言](https://www.hiascend.com/zh/ascend-c)开发的昇腾基础算子，源码目录`src`下按照不同算子类型进行分类：

```
src  // 算子源码目录
  ├── activation    // 激活函数类算子
  ├── common        // 公共目录
  ├── conv          // 卷积类算子
  ├── conversion    // 张量变换类算子
  ├── contrib       // 社区贡献算子目录✨
  ├── foreach       // foreach类算子
  ├── image         // 图像处理类算子
  ├── index         // 索引计算类算子
  ├── loss          // 损失函数类算子
  ├── math          // 数学计算类算子
  ├── matmul        // 矩阵计算类算子
  ├── norm          // 正则化类算子
  ├── objdetect     // 目标检测类算子
  ├── optim         // 优化器类算子
  ├── pooling       // 池化类算子
  ├── quant         // 量化&反量化类算子
  ├── random        // 随机数类算子
  └── rnn           // 循环神经网络类算子
```

上面各分类目录中存放了昇腾对外开源的AscendC算子，用户可以在CANN软件包中使用。

`社区贡献算子目录✨`用来存放实验性算子和社区开源贡献的算子，内部的算子目录结构与上面各算子分类相同。当开源贡献算子成熟后会被纳入CANN软件包管理并移动到外层对应分类目录。

各分类下每个算子的目录结构类似，可以看做一个[msopgen](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0018.html)算子工程。以`add_custom`为例，其目录结构如下：

```
add_custom/
  ├── CMakeLists.txt            // CMake构建配置文件
  ├── docs                      // 算子文档目录
  │   └── AddCustom.md          // 算子aclnn接口文档
  ├── examples                  // 调用示例目录
  │   ├── AclNNInvocationNaive  // aclnn接口调用示例
  │   ├── PytorchInvocation     // pytorch接口调用示例（可选）
  │   ├── TensorflowInvocation  // tf接口调用示例（可选）
  │   ...
  ├── framework                 // 第三方框架适配目录（可选）
  │   ├── CMakeLists.txt
  │   └── tf_plugin             // TensorFlow适配代码
  ├── op_host                   // host侧文件目录
  │   ├── add_custom.cpp        // tiling实现与算子信息库配置
  │   └── add_custom_tiling.h   // tilingData数据定义
  ├── op_kernel                 // kernel侧文件目录
  │   └── add_custom.cpp        // 算子实现文件
  ├── opp_kernel_aicpu          // aicpu实现目录（可选）
  ├── README.md                 // 算子介绍文档
  └── tests                     // 算子测试用例
      ├── st                    // 系统测试用例
      └── ut                    // 单元测试用例（可选）
```

开发者可参考仓库中已有算子代码结构，按需修改和添加实现文件。

## 3. 环境准备

详细安装步骤请参考[CANN社区版文档-环境准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/softwareinst/instg/instg_0001.html)相关章节，对昇腾硬件、CANN软件及相应深度学习框架进行安装准备。

### 3.1 软件依赖

以下所列仅为cann-ops源码编译用到的依赖，其中python、gcc的安装方法请参见配套版本的[用户手册](https://hiascend.com/document/redirect/CannCommunityInstDepend)，选择安装场景后，参见“安装CANN > 安装依赖”章节进行相关依赖的安装。

相关软件依赖版本要求如下：

   - python >= 3.7.0

   - gcc >= 7.3.0

   - cmake >= 3.16.0

   - protobuf <=3.20.x

     算子编译时，protobuf版本需低于3.20.x，您可以执行**pip3 list**命令查询当前环境中的protobuf版本，如果版本高于3.20.x，则执行如下命令重新安装，以重新安装3.20.0版本为例：

     ```bash
     pip3 install protobuf==3.20.0
     ```

     如果使用非root用户安装，需要在安装命令后加上--user，例如**pip3 install protobuf==3.20.0 --user**。

   - googletest（可选，仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

     如下以[googletest源码](https://github.com/google/googletest.git)编译安装为例，安装命令如下：

     ```bash
     git checkout release-1.11.0          # 切换到googletest项目的 release-1.11.0 tag版本
     mkdir temp && cd temp                # 在googletest源码根目录下创建临时目录并进入
     cmake .. -DCMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"
     make
     make install                         # root用户安装googletest
     # sudo make install                  # 非root用户安装googletest
     ```

   - nlohmann_json (建议版本 [release-3.11.3](https://github.com/nlohmann/json/releases/tag/v3.11.3))

     如下以[json源码](https://github.com/nlohmann/json.git)编译安装为例，安装命令如下：

     ```bash
     git checkout v3.11.3                 # 切换到json项目的 v3.11.3 tag版本
     mkdir build && cd build              # 在json源码根目录下创建构建目录并进入
     cmake .. -DJSON_BuildTests=OFF       # 禁用测试以加快构建
     cmake --install .                    # root用户安装，默认安装到系统路径（/usr/local)
     # sudo cmake --install .             # 非root用户安装
     ```

### 3.2 CANN开发套件包安装及路径信息

   执行安装命令时，请确保安装用户对软件包具有可执行权限。

   - 使用默认路径安装

     ```bash
     # CANN开发套件包安装命令示例：
     ./Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install
     # 算子二进制包安装命令示例：
     ./Ascend-cann-kernels-<soc_version>_<cann_version>_linux.run --install
     ```

     - 若使用root用户安装，安装完成后CANN开发套件包存储在`/usr/local/Ascend/ascend-toolkit/latest`路径；算子二进制包存储在`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel`路径。
     - 若使用非root用户安装，安装完成后CANN开发套件包存储在`$HOME/Ascend/ascend-toolkit/latest`路径；算子二进制包存储在`${HOME}/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel`路径。

   - 指定路径安装

     ```bash
     # CANN开发套件包安装命令示例：
     ./Ascend-cann-toolkit_<cann_version>_linux-<arch>.run --install --install-path=${install_path}
     # 算子二进制包安装命令示例：
     ./Ascend-cann-kernels-<soc_version>_<cann_version>_linux.run --install --install-path=${install_path}
     ```

     安装完成后，CANN开发套件包存储在\${install_path}指定路径；算子二进制包存储在`${install_path}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel`路径。

### 3.3 设置环境变量

   - 默认路径，root用户安装

     ```bash
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     ```

   - 默认路径，非root用户安装

     ```bash
     source $HOME/Ascend/ascend-toolkit/set_env.sh
     ```

   - 指定路径安装

     ```bash
     source ${install_path}/ascend-toolkit/set_env.sh
     ```

   **注意**：
   1. 若环境中已安装多个版本的CANN软件包，设置上述环境变量时，请确保${install_path}/ascend-toolkit/latest目录指向的是配套版本的软件包。
   2. 当用户选择将CANN开发套件包安装在自定义指定路径下时，后续算子调用示例中使用的环境变量需要修改为对应的指定路径
      ```bash
      export DDK_PATH=${install_path}/ascend-toolkit/latest
      export NPU_HOST_LIB=${install_path}/ascend-toolkit/latest/lib64
      ```

## 4. 源码下载

1. 通过`git`命令下载本仓源码：

   ```bash
   git clone https://gitee.com/ascend/cann-ops.git
   ```

2. 通过gitee网页下载本仓源码zip压缩包:
  
   开发者可以点击[cann-ops项目主页](https://gitee.com/ascend/cann-ops)的“克隆/下载”按钮，然后点击“下载ZIP”选项获取项目压缩包，之后解压查看项目代码。

   **注意**：如果用户在windows系统中下载并解压代码，然后再将代码上传到Linux服务器，由于系统文件权限差异，可能导致项目中部分脚本失去可执行权限。建议将压缩包上传到Linux服务器后再使用`unzip`工具进行解压。

## 5. 编译执行

仓库中提供了编译脚本`build.sh`，进入本仓代码根目录，可执行命令如下：

  ```bash
  bash build.sh                     # 自定义算子包编译
  bash build.sh -n <算子名称>        # 编译指定算子出包
  bash build.sh -t                  # 单元测试编译执行
  ```

您还可以通过`bash build.sh --help`命令查询更多可用参数；执行完自定义算子包编译命令，需要完成[自定义算子包安装](#52-自定义算子包安装)后，才能执行后续命令。

### 5.1 自定义算子包编译

进入本仓代码根目录，根据开发者使用的昇腾芯片类型和CANN开发套件包安装路径修改`CMakePresets.json`配置文件中的`ASCEND_COMPUTE_UNIT`和`ASCEND_CANN_PACKAGE_PATH`选项值。

```json
...
"ASCEND_COMPUTE_UNIT": {
    "type": "STRING",
    "value": "ascendxxxy"
},
...
"ASCEND_CANN_PACKAGE_PATH": {
    "type": "PATH",
    "value": "/usr/local/Ascend/ascend-toolkit/latest"
}
...
```

其中，`ASCEND_COMPUTE_UNIT`的值`ascendxxxy`可通过`npu-smi info`命令进行查询，显示的`Name`列即为使用的芯片类型；`ASCEND_CANN_PACKAGE_PATH`的值可通过`echo ${ASCEND_HOME_PATH}`进行查询。

然后，执行以下命令对项目进行编译：
```bash
bash build.sh
```

**注意**：编译时项目会检查用户环境是否安装了所需的依赖包，如果缺少相关依赖，项目会通过网络进行下载，请确认当前机器可以正常访问互联网。

若提示如下信息，则说明编译成功。

```plain text
Self-extractable archive "CANN-custom_ops-<cann_version>-linux.<arch>.run" successfully created.
```

编译成功后在 `本仓代码根目录/build_out` 目录生成自定义算子包：`CANN-custom_ops-<cann_version>-linux.<arch>.run`。

其中，\<cann_version>表示软件版本号，\<arch>表示操作系统架构。

### 5.2 自定义算子包安装

安装前，需确保所安装的自定义算子包与所安装CANN开发套件包CPU架构一致，并且要先设置CANN开发套件包环境变量，然后再进行安装，仅支持在配套版本安装自定义算子包，安装命令如下：

  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh # 设置CANN开发套件包环境变量，以root用户默认路径为例，如已设置，则请忽略该操作
  ./build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run --quiet # 安装自定义算子run包
  ```

执行上述命令后，自定义算子run包会默认安装到CANN软件包目录，例如，`/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/` 目录。

如果用户想将自定义算子包安装到自己的指定路径，可通过`--install-path`参数来指定：

  ```bash
  ./build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run --quiet --install-path=${install_path}
  ```

安装成功后，根据屏幕上的提示信息更新`LD_LIBRARY_PATH`环境变量，以安装到root用户默认路径为例，执行命令为`export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH}`。

> 更多自定义算子包相关功能可通过执行`./build_out/CANN-custom_ops-<cann_version>-linux.<arch>.run --help`查看。

### 5.3 示例工程编译执行

此操作需要在真实NPU环境上进行，并且依赖CANN开发套件包和算子二进制包，因此在编译前，需要参见[环境准备](#3-环境准备)章节安装配套版本的CANN开发套件包和算子二进制包，并设置环境变量。然后安装好编译出的自定义算子包并设置环境变量。

#### 5.3.1 aclnn接口调用验证

进入需要执行的算子的`examples`目录，选择希望的执行方式，大多数算子会提供aclnn接口的调用方式，按照`AclNNInvocationNaive/README.md`文档中的步骤可以手动编译执行用例，也可以执行`run.sh`一键式编译验证脚本。

aclnn验证工程的结构如下：

```
AclNNInvocationNaive
    ├── CMakeLists.txt      // CMake构建配置文件
    ├── gen_data.py         // 输入数据生成与标杆结果计算脚本
    ├── main.cpp            // aclnn接口调用实现
    ├── README.md           // README文档
    ├── run.sh              // 一键式验证脚本
    └── verify_result.py    // 结果验证脚本
```

aclnn接口调用验证的主要流程为：

1. 调用`gen_data.py`脚本生成对应shape和范围的输入数据，计算标杆结果，将输入数据和标杆结果保存为`.bin`文件
2. 编译`main.cpp`文件生成aclnn接口调用的可执行文件`execute_test_op`，调用并保存aclnn接口的输出结果
3. 调用`verify_result.py`脚本对标杆结果和aclnn结果数据进行比较，判断算子计算结果是否符合预期

> 更多关于aclnn接口编译运行的信息可参考[编译与运行样例](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/aolapi/context/common/%E7%BC%96%E8%AF%91%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%A0%B7%E4%BE%8B.md)文档。

#### 5.3.2 ST测试验证

进入需要执行的算子的`tests/st`目录，按照`README.md`文档进行环境配置，然后通过`msopst`工具执行ST测试用例验证。

ST验证工程的结构如下（以add_custom为例）：

```
st
  ├── AddCustom_case_all_type.json  // ST测试用例
  ├── msopst.ini                    // ST测试配置文件（可选）
  ├── README.md                     // README文档
  └── test_add_custom.py            // 算子标杆函数脚本（可选），函数名需要定义为calc_expect_func
```

msopst工具会根据json测试用例中配置的shape、dtype和数据范围等信息生成对应的数据调用算子kernel进行功能验证。如果提供了算子标杆函数脚本，msopst还会对算子输出结果和标杆计算结果进行精度比较。

> 更多关于ST测试的信息可参考[算子测试msOpSt](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/ODtools/Operatordevelopmenttools/atlasopdev_16_0029.html)工具文档。

## 6. 开发指导

### 6.1 对仓库中的算子进行修改

当用户对仓库中已有算子进行功能增强、性能优化等自定义修改后，为了加快编译出包速度，可以在编译时通过`-n`参数指定算子名称，仅编译自己修改的算子，以add_custom算子为例：

```bash
bash build.sh -n add_custom
```

此时，得到的run包就会只包含add_custom算子实现，减少编译等待时间。run包的安装方式与[自定义算子包安装](#52-自定义算子包安装)相同。

### 6.2 新增自定义算子

因为仓库中各个算子的目录结构是类似的，用户可以复制已有算子的目录结构进行修改，添加算子`op_host`、`op_kernel`实现代码, 按需添加ST或者aclnn接口测试用例。执行单算子编译出包`bash build.sh -n <算子名称>`然后安装验证。

> 详细的修改验证过程可进一步参考[本地构建和验证模式一](docs/contributors/build-verf-mode1.md)或[本地构建和验证模式二](docs/contributors/build-verf-mode2.md)文档。

## 7. 贡献指南

cann-ops仓欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[Contributing](docs/Contributing.md)了解行为准则，进行CLA协议签署，以及参与源码仓贡献的详细流程。

**针对cann-ops仓，开发者准备本地代码与提交PR时需要重点关注如下几点**：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。

2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。

## 许可证

[CANN Open Software License Agreement Version 1.0](LICENSE)
