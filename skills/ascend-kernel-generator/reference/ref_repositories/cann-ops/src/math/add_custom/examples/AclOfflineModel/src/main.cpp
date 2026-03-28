/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file main.cpp
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"

bool g_isDevice = false;
int g_isDynamic = 0;
int g_length = 0;
const int DYNAMIC_OP_ARGS_COUNT = 3;
const int STATIC_OP_ARGS_COUNT = 1;
const int IS_DYN_IDX = 1;
const int LENGTH_IDX = 2;
const int DEVICE_ID = 0;

OperatorDesc CreateOpDesc()
{
    // define operator
    std::vector<int64_t> shape{8, 2048};
    std::string opType = "AddCustom";
    if (g_isDynamic) {
        shape = {8, g_length};
    }
    aclDataType dataType = ACL_FLOAT16;
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc(opType);
    opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    opDesc.AddOutputTensorDesc(dataType, shape.size(), shape.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    ReadFile("../input/input_x.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("../input/input_y.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    WriteFile("../output/output_z.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    INFO_LOG("Write output success");
    return true;
}

bool RunOp()
{
    // Create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // Create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    // Process output data
    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

void DestroyResource()
{
    bool flag = false;
    if (aclrtResetDevice(DEVICE_ID) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", DEVICE_ID);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destroy resource failed");
    } else {
        INFO_LOG("Destroy resource success");
    }
}

bool InitResource()
{
    std::string output = "./output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    // acl.json is dump or profiling config file
    if (aclInit("../scripts/acl.json") != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(DEVICE_ID) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. DEVICE_ID is %d", DEVICE_ID);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", DEVICE_ID);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    // set model path
    if (aclopSetModelDir("../op_models") != ACL_SUCCESS) {
        std::cerr << "Load single op model failed" << std::endl;
        (void)aclFinalize();
        return FAILED;
    }
    INFO_LOG("aclopSetModelDir op model success");

    return true;
}

int main(int argc, char **argv)
{
    if (argc == ARGC_NUM) {
        INFO_LOG("dynamic op will be called");
        g_isDynamic = atoi(argv[IS_DYN_IDX]);
        g_length = atoi(argv[LENGTH_IDX]);
    } else if (argc == 1) {
        INFO_LOG("static op will be called");
    } else {
        ERROR_LOG("wrong input parameter number");
        return -1;
    }

    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestroyResource();
        return FAILED;
    }

    DestroyResource();

    return SUCCESS;
}
