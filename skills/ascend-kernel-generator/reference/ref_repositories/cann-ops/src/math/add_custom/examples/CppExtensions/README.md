## 使用cpp-extension的方式调用AddCustom算子工程

cpp-extension方式是通过编译出一个C++的算子扩展包的形式来调用，可以分为Jit（即时编译）、load_library、编译wheel包和自定义算子入图注册（可选）的形式。以下均使用Pytorch2.1.0完成样例验证，支持Python3.8和Python3.9。
样例中各自包含run.sh执行脚本供用户参考使用。

### 编译前准备

- 安装pytorch和torch-npu

根据[torch-npu](https://gitee.com/ascend/pytorch)的安装说明，进行源码编译安装或直接安装torch-npu包。以Pytorch2.1.0、Python3.9、CANN版本8.0.0.alpha003为例，安装命令如下，具体操作或者其他版本pytorch安装请参照[torch-npu](https://gitee.com/ascend/pytorch)源码仓，此处仅为示例。
```
  pip3 install torch==2.1.0
  pip3 install torch-npu==2.1.0.post10
```

### 使用Jit的方式调用

该样例脚本基于Pytorch2.1运行，2.1以下版本中NPU设备绑定的设备名称有变化，可自行参考样例代码中的注释说明。

#### 安装依赖

- 安装编译依赖
  
  ```bash
  pip3 install pyyaml
  pip3 install wheel
  pip3 install setuptools
  ```
  
  - 安装测试依赖
    ```bash
    pip3 install Ninja
    pip3 install expecttest
    ```

#### 样例运行

- 进入到样例目录
  以命令行方式下载样例代码，master分支为例。
  
  ```bash
  cd ${git_clone_path}/cann-ops/src/math/add_custom/examples/CppExtensions/jit
  ```
  
  - 配置环境变量
    
    ```bash
    export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
    ```
  - 样例执行
    
    样例执行过程中会自动生成测试数据，然后通过cpp_extension扩展机制，使用实时编译模式编译并执行AclNN算子接口，最后检验运行结果。
    
    ```bash
    cd test
    python3 test_add_custom.py
    ```
    
    用户亦可参考run.sh脚本进行编译与运行。
    
    ```bash
    bash run.sh
    ```

### 使用load_library的方式调用

该样例基于Pytorch2.1运行，2.1以下版本中NPU设备绑定的设备名称有变化，可自行参考样例代码中的注释说明。

#### 编译自定义API的so

- 进入到样例目录
  以命令行方式下载样例代码，master分支为例。
  
  ```bash
  cd ${git_clone_path}/cann-ops/src/math/add_custom/examples/CppExtensions/load_library
  ```
  
  - 执行编译命令
    
    ```bash
    mkdir build
    cd build
    cmake ..
    make -j
    ```

#### 样例运行

- 进入到样例目录
  以命令行方式下载样例代码，master分支为例。
  
  ```bash
  cd ${git_clone_path}/cann-ops/src/math/add_custom/examples/CppExtensions/load_library/test
  ```
  
  - 配置环境变量
    
    ```bash
    export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
    ```
  - 样例执行
    
    ```bash
    python3 test_add_custom.py
    ```
    
    用户亦可参考run.sh脚本进行编译与运行。
    
    ```bash
    bash run.sh
    ```

### 使用编译wheel包的方式调用

该样例基于Pytorch2.1运行，2.1以下版本中NPU设备绑定的设备名称有变化，可自行参考样例代码中的注释说明。

#### 编译自定义算子wheel包

- 进入到样例目录
  以命令行方式下载样例代码，master分支为例。
  
  ```bash
  cd ${git_clone_path}/cann-ops/src/math/add_custom/examples/CppExtensions/setup
  ```
  
  - 执行编译命令
    
    ```bash
    python3 setup.py build bdist_wheel
    ```
  - 安装wheel包
    
    ```bash
    cd dist/
    pip3 install custom_ops-1.0-cp38-cp38-linux_aarch64.whl (需要修改为实际编译出的whl包)
    ```

#### 样例运行

- 进入到样例目录
  以命令行方式下载样例代码，master分支为例。
  
  ```bash
  cd ${git_clone_path}/cann-ops/src/math/add_custom/examples/CppExtensions/setup/test
  ```
  
  - 配置环境变量
    
    ```bash
    export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
    ```
  - 样例执行
    
    ```bash
    python3 test_add_custom.py
    ```
    
    用户亦可参考run.sh脚本进行编译与运行。
    
    ```bash
    bash run.sh
    ```

#### 自定义算子入图注册（可选）

此功能可以让用户将开发的自定义算子增加入图能力，为可选步骤，如不需要入图则可以跳过。进行此步骤需要先安装[torchair](https://gitee.com/ascend/torchair)（当前仅支持Pytorch2.1），然后为自定义算子注册meta后端实现，用来完成图模式下的shape推导，具体参考extension_add.cpp文件中为Meta设备注册后端实现的相关代码，示例代码中相关注册过程已经添加，按如下步骤进行编译注册等即可完成入图。

- 根据Ascend C工程产生的REG_OP算子原型填充torchair.ge.custom_op的参数。
  AddCustom的REG_OP原型为：
  
  ```cpp
  REG_OP(AddCustom)
      .INPUT(x, ge::TensorType::ALL())
      .INPUT(y, ge::TensorType::ALL())
      .OUTPUT(z, ge::TensorType::ALL())
      .OP_END_FACTORY_REG(AddCustom);
  ```
  
  - 注册自定义算子的converter
    
    在自己的调用文件里面调用@register_fx_node_ge_converter，完成converter注册，这部分代码在test/test_add_custom_graph.py用例里面已经添加。
    
    ```python
    from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable
    import torchair
    from torchair import register_fx_node_ge_converter
    from torchair.ge import Tensor
    import custom_ops
    
    
    # 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
    @register_fx_node_ge_converter(torch.ops.myops.my_op.default)
    def convert_npu_add_custom(x: Tensor, y: Tensor, z: Tensor = None, meta_outputs: Any = None):
        return torchair.ge.custom_op(
            "AddCustom",
            inputs={
                "x": x,
                "y": y,
            },
            outputs=['z']
        )
    ```
  - 样例执行
    
    ```bash
    cd ${git_clone_path}/cann-ops/src/math/add_custom/examples/CppExtensions/setup/test
    python3 test_add_custom_graph.py
    ```

## 更新说明

| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/01/06 | 新增本readme |
