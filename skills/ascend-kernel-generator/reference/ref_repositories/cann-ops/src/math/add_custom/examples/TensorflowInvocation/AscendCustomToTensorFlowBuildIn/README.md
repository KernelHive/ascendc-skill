## 概述
本样例展示了如何使用Ascend C自定义算子AddCustom映射到TensorFlow自定义算子Add，并通过TensorFlow调用Ascend C算子。

## 运行样例算子
### 1.编译部署自定义算子
参考[编译算子工程](../../../README.md#编译部署自定义算子)。
注意：若tensorflow版本是2.x，需注意插件代码适配，路径为： samples/operator/AddCustomSample/FrameworkLaunch/AddCustom/framework/tf_plugin/tensorflow_add_custom_plugin.cc
需修改插件代码src/math/add_custom/framework/tf_plugin/tensorflow_add_custom_plugin.cc中的TensorFlow调用算子名称OriginOpType为"AddV2"，如下所示：
```c++
REGISTER_CUSTOM_OP("AddCustom")
  .FrameworkType(TENSORFLOW)      // type: TENSORFLOW
  .OriginOpType("AddV2")      // name in tf module
  .ParseParamsByOperatorFn(AutoMappingByOpFn);
```
还需要修改src/math/add_custom/CMakeLists.txt，在结尾加上一行：
```
add_subdirectory(framework)
```
### 2.tensorflow调用的方式调用样例运行

  - 进入到样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/AddCustomSample/FrameworkLaunch/TensorflowInvocation/AscendCustomToTensorFlowBuildIn
    ```

  - 样例执行(tensorflow1.15)

    样例执行过程中会自动生成随机测试数据，然后通过TensorFlow调用算子，最后对比cpu和aicore运行结果。具体过程可参见run_add_custom.py脚本。
    ```bash
    python3 run_add_custom.py
    ```
  - 样例执行(tensorflow2.x)

    样例执行过程中会自动生成随机测试数据，然后通过TensorFlow调用算子，最后对比cpu和aicore运行结果。具体过程可参见run_add_custom_tf2.py脚本。
    ```bash
    python3 run_add_custom_tf2.py
    ```

    用户亦可参考run.sh脚本进行编译与运行。
    ```bash
    bash run.sh
    ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/01/06 | 新增本readme |