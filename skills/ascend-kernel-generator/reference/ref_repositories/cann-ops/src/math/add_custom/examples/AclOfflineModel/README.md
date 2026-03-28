## 使用aclopExecuteV2模型调用的方式调用AddCustom算子工程
该样例暂不支持Atlas 200/500 A2 推理产品。

## 运行样例算子
  **请确保已根据算子包编译部署步骤完成本算子的编译部署动作。**
  
  - 进入到样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/cann-ops/src/math/add_custom/examples/AclOfflineModel
    ```

  - 样例执行

    样例执行过程中会自动生成测试数据，然后编译与运行acl离线模型调用样例，最后检验运行结果。具体过程可参见run.sh脚本。
    ```bash
    bash run.sh
    ```
## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/01/06 | 新增本readme |