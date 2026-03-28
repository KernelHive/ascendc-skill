## 概述
本样例展示了如何使用Ascend C自定义算子GatherV3映射到TensorFlow自定义算子GatherV3，并通过TensorFlow调用Ascend C算子。

## 运行样例算子
### 1.编译部署自定义算子
参考[编译算子工程](../../../README.md#编译部署自定义算子)。
需修改插件代码cann-ops/src/index/gather_v3/framework/GatherV3/framework/tf_plugin/tensorflow_gather_v3_plugin.cc中的TensorFlow调用算子名称OriginOpType为"GatherV3"，如下所示：
```c++
REGISTER_CUSTOM_OP("GatherV3")
  .FrameworkType(TENSORFLOW)      // type: CAFFE, TENSORFLOW
  .OriginOpType("GatherV3")      // name in tf module
  .ParseParamsByOperatorFn(AutoMappingByOpFn);
```

### 2.TensorFlow调用的方式调用样例运行

  - 进入到样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/src/index/gather_v3/examples/TensorflowInvocation
    ```
  - 编译TensorFlow算子库
    ```bash
    bash build.sh
    ```

  - 样例执行(TensorFlow1.15)

    样例执行过程中会自动生成随机测试数据，然后通过TensorFlow调用算子，最后对比TensorFlow原生算子和Ascend C算子运行结果。具体过程可参见run_gather_v3_tf_1_15.py脚本。
    ```bash
    python3 run_gather_v3_tf_1_15.py
    ```
  - 样例执行(TensorFlow2.6.5)
    样例执行过程中会自动生成随机测试数据，然后通过TensorFlow调用算子，最后对比TensorFlow原生算子和Ascend C算子运行结果。具体过程可参见run_gather_v3_tf_2_6_5.py脚本。
    ```bash
    python3 run_gather_v3_tf_2_6_5.py
    ```


## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/04/10 | 新增本readme及样例 |