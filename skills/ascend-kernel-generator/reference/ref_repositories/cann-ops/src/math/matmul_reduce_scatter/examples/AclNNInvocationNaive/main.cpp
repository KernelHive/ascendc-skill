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

#include <cstdint>
#include <iostream>
#include <thread>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>
#include <limits>
#include <cassert>
#include <vector>
#include <string>
#include <iomanip>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"
#include "aclnn_matmul_reduce_scatter.h"

#ifndef MATMUL_REDUCE_SCATTER_DEMO_DEF_H
#define MATMUL_REDUCE_SCATTER_DEMO_DEF_H

constexpr uint32_t RANK_DIM = 1;
constexpr uint32_t RANK_M = 16384;
constexpr uint32_t RANK_K = 640;
constexpr uint32_t RANK_N = 5120;
constexpr bool IS_TRANS_A = false;
constexpr bool IS_TRANS_B = false;
constexpr int64_t COMM_TURN = 0;
constexpr char REDUCE_OP[] = "sum";

#endif

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

bool g_isDevice = false;

// Original common.h and common.cpp content
bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        WARN_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void) close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

// Original operator_desc.h and operator_desc.cpp content
struct OperatorDesc {
    OperatorDesc();
    virtual ~OperatorDesc();

    OperatorDesc &AddInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);
    OperatorDesc &AddOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

    std::string opType;
    std::vector<aclTensorDesc *> inputDesc;
    std::vector<aclTensorDesc *> outputDesc;
};

OperatorDesc::OperatorDesc() {}

OperatorDesc::~OperatorDesc()
{
    for (auto *desc : inputDesc) {
        aclDestroyTensorDesc(desc);
    }

    for (auto *desc : outputDesc) {
        aclDestroyTensorDesc(desc);
    }
}

OperatorDesc &OperatorDesc::AddInputTensorDesc(aclDataType dataType,
                                               int numDims,
                                               const int64_t *dims,
                                               aclFormat format)
{
    aclTensorDesc *desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create tensor failed");
        return *this;
    }
    inputDesc.emplace_back(desc);
    return *this;
}

OperatorDesc &OperatorDesc::AddOutputTensorDesc(aclDataType dataType,
                                                int numDims,
                                                const int64_t *dims,
                                                aclFormat format)
{
    aclTensorDesc *desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create tensor failed");
        return *this;
    }

    outputDesc.emplace_back(desc);
    return *this;
}

// Original op_runner.h and op_runner.cpp content
class OpRunner {
public:
    explicit OpRunner(OperatorDesc *opDesc);
    virtual ~OpRunner();

    bool Init();
    const size_t NumInputs();
    const size_t NumOutputs();
    const size_t GetInputSize(size_t index) const;
    const size_t GetInputNumDims(size_t index) const;
    aclDataType GetInputDataType(size_t index) const;
    aclFormat GetInputFormat(size_t index) const;
    size_t GetOutputSize(size_t index) const;
    const size_t GetOutputNumDims(size_t index) const;
    aclDataType GetOutputDataType(size_t index) const;
    aclFormat GetOutputFormat(size_t index) const;
    size_t GetInputElementCount(size_t index) const;
    size_t GetOutputElementCount(size_t index) const;
    std::vector<int64_t> GetInputShape(size_t index) const;
    std::vector<int64_t> GetOutputShape(size_t index) const;
    std::vector<int64_t> GetShapeFromTensorDesc(aclTensorDesc* desc) const;

    template<typename T>
    auto GetInputBuffer(size_t index) -> T*
    {
        if (index >= numInputs_) {
            ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
            return nullptr;
        }
        return reinterpret_cast<T *>(hostInputs_[index]);
    }

    template<typename T>
    auto GetOutputBuffer(size_t index) -> const T*
    {
        if (index >= numOutputs_) {
            ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
            return nullptr;
        }
        return reinterpret_cast<const T *>(hostOutputs_[index]);
    }

    void PrintInput(size_t index, size_t elementsPerRow = 16);
    void PrintOutput(size_t index, size_t elementsPerRow = 16);
    bool CompileStaticOp();
    bool CompileDynamicOp();
    bool RunOp(std::string group, aclrtStream stream);

private:
    uint32_t rankSize_ { 0 };
    size_t numInputs_ { 0 };
    size_t numOutputs_ { 0 };

    std::vector<aclDataBuffer *> inputBuffers_;
    std::vector<aclDataBuffer *> outputBuffers_;

    std::vector<void *> devInputs_;
    std::vector<void *> devOutputs_;

    std::vector<void *> hostInputs_;
    std::vector<void *> hostOutputs_;

    std::vector<aclTensor *> inputTensor_;
    std::vector<aclTensor *> outputTensor_;
    OperatorDesc *opDesc_;
};

// Implementation of OpRunner methods...
OpRunner::OpRunner(OperatorDesc *opDesc) : opDesc_(opDesc)
{
    numInputs_ = opDesc->inputDesc.size();
    numOutputs_ = opDesc->outputDesc.size();
}

OpRunner::~OpRunner()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        (void)aclDestroyTensor(inputTensor_[i]);
        (void)aclDestroyDataBuffer(inputBuffers_[i]);
        (void)aclrtFree(devInputs_[i]);
        if (g_isDevice) {
            (void)aclrtFree(hostInputs_[i]);
        } else {
            (void)aclrtFreeHost(hostInputs_[i]);
        }
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        (void)aclDestroyTensor(outputTensor_[i]);
        (void)aclDestroyDataBuffer(outputBuffers_[i]);
        (void)aclrtFree(devOutputs_[i]);
        if (g_isDevice) {
            (void)aclrtFree(hostOutputs_[i]);
        } else {
            (void)aclrtFreeHost(hostOutputs_[i]);
        }
    }
}

bool OpRunner::Init()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        auto size = GetInputSize(i);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for input[%zu] failed", i);
            return false;
        }
        devInputs_.emplace_back(devMem);
        inputBuffers_.emplace_back(aclCreateDataBuffer(devMem, size));

        void *hostInput = nullptr;
        if (g_isDevice) {
            if (aclrtMalloc(&hostInput, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                return false;
            }
        } else {
            if (aclrtMallocHost(&hostInput, size) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                return false;
            }
        }
        if (hostInput == nullptr) {
            ERROR_LOG("Malloc memory for input[%zu] failed", i);
            return false;
        }
        hostInputs_.emplace_back(hostInput);

        aclTensor *inputTensor = aclCreateTensor(GetInputShape(i).data(), GetInputNumDims(i), GetInputDataType(i),
            nullptr, 0, GetInputFormat(i), GetInputShape(i).data(), GetInputNumDims(i), devInputs_[i]);
        if (inputTensor == nullptr) {
            ERROR_LOG("Create Tensor for input[%zu] failed", i);
            return false;
        }
        inputTensor_.emplace_back(inputTensor);
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        auto size = GetOutputSize(i);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for output[%zu] failed", i);
            return false;
        }
        devOutputs_.emplace_back(devMem);
        outputBuffers_.emplace_back(aclCreateDataBuffer(devMem, size));

        void *hostOutput = nullptr;
        if (g_isDevice) {
            if (aclrtMalloc(&hostOutput, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for output[%zu] failed", i);
                return false;
            }
        } else {
            if (aclrtMallocHost(&hostOutput, size) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for output[%zu] failed", i);
                return false;
            }
        }
        if (hostOutput == nullptr) {
            ERROR_LOG("Malloc host memory for output[%zu] failed", i);
            return false;
        }
        hostOutputs_.emplace_back(hostOutput);

        aclTensor *outputTensor = aclCreateTensor(GetOutputShape(i).data(), GetOutputNumDims(i), GetOutputDataType(i),
            nullptr, 0, GetOutputFormat(i), GetOutputShape(i).data(), GetOutputNumDims(i), devOutputs_[i]);
        if (outputTensor == nullptr) {
            ERROR_LOG("Create Tensor for output[%zu] failed", i);
            return false;
        }
        outputTensor_.emplace_back(outputTensor);
    }

    return true;
}

const size_t OpRunner::NumInputs()
{
    return numInputs_;
}

const size_t OpRunner::NumOutputs()
{
    return numOutputs_;
}

const size_t OpRunner::GetInputSize(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescSize(opDesc_->inputDesc[index]);
}

const size_t OpRunner::GetInputNumDims(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescNumDims(opDesc_->inputDesc[index]);
}

aclDataType OpRunner::GetInputDataType(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ACL_DT_UNDEFINED;
    }

    return aclGetTensorDescType(opDesc_->inputDesc[index]);
}

aclFormat OpRunner::GetInputFormat(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ACL_FORMAT_UNDEFINED;
    }

    return aclGetTensorDescFormat(opDesc_->inputDesc[index]);
}

std::vector<int64_t> OpRunner::GetShapeFromTensorDesc(aclTensorDesc* desc) const {
    std::vector<int64_t> shape;
    if (desc == nullptr) {
        ERROR_LOG("tensor description is null");
        return shape;
    }

    size_t dims = aclGetTensorDescNumDims(desc);
    for (size_t i = 0; i < dims; ++i) {
        int64_t dimSize;
        if (aclGetTensorDescDimV2(desc, i, &dimSize) != ACL_SUCCESS) {
            ERROR_LOG("get dims from tensor desc failed. dims index = %zu", i);
            shape.clear();
            return shape;
        }
        shape.emplace_back(dimSize);
    }
    return shape;
}

std::vector<int64_t> OpRunner::GetInputShape(size_t index) const {
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return {};
    }
    return GetShapeFromTensorDesc(opDesc_->inputDesc[index]);
}

std::vector<int64_t> OpRunner::GetOutputShape(size_t index) const {
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return {};
    }
    return GetShapeFromTensorDesc(opDesc_->outputDesc[index]);
}

size_t OpRunner::GetOutputSize(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescSize(opDesc_->outputDesc[index]);
}

const size_t OpRunner::GetOutputNumDims(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescNumDims(opDesc_->outputDesc[index]);
}

aclDataType OpRunner::GetOutputDataType(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ACL_DT_UNDEFINED;
    }

    return aclGetTensorDescType(opDesc_->outputDesc[index]);
}


aclFormat OpRunner::GetOutputFormat(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ACL_FORMAT_UNDEFINED;
    }

    return aclGetTensorDescFormat(opDesc_->outputDesc[index]);
}

size_t OpRunner::GetInputElementCount(size_t index) const
{
    if (index >= opDesc_->inputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescElementCount(opDesc_->inputDesc[index]);
}

size_t OpRunner::GetOutputElementCount(size_t index) const
{
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescElementCount(opDesc_->outputDesc[index]);
}

bool OpRunner::RunOp(std::string group, aclrtStream stream)
{
    for (size_t i = 0; i < numInputs_; ++i) {
        auto size = GetInputSize(i);
        aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_DEVICE;
        if (g_isDevice) {
            kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
        }
        if (aclrtMemcpy(devInputs_[i], size, hostInputs_[i], size, kind) != ACL_SUCCESS) {
            ERROR_LOG("Copy input[%zu] failed", i);
            return false;
        }
        INFO_LOG("Copy input[%zu] success", i);
    }

    aclTensor *bias = nullptr;
    size_t workspaceSize = 0;
    aclOpExecutor *handle = nullptr;
    auto ret = aclnnMatmulReduceScatterGetWorkspaceSize(inputTensor_[0], inputTensor_[1], bias, (char*)group.c_str(),
        const_cast<char*>(REDUCE_OP), IS_TRANS_A, IS_TRANS_B, COMM_TURN, outputTensor_[0], &workspaceSize, &handle);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Get Operator Workspace failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("Execute aclnnMatmulReduceScatterGetWorkspaceSize success, workspace size %lu", workspaceSize);
    
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory failed");
        }
    }

    ret = aclnnMatmulReduceScatter(workspace, workspaceSize, handle, stream);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Execute Operator failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("Execute aclnnMatmulReduceScatter success");

    ret = aclrtSynchronizeStreamWithTimeout(stream, 10000); // 流同步超时10000 
    if (ret != SUCCESS) {
        ERROR_LOG("Synchronize stream failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    INFO_LOG("Synchronize stream success");

    for (size_t i = 0; i < numOutputs_; ++i) {
        auto size = GetOutputSize(i);
        aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
        if (g_isDevice) {
            kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
        }
        if (aclrtMemcpy(hostOutputs_[i], size, devOutputs_[i], size, kind) != ACL_SUCCESS) {
            INFO_LOG("Copy output[%zu] success", i);
            return false;
        }
        INFO_LOG("Copy output[%zu] success", i);
    }
    return true;
}


template<typename T>
void DoPrintData(const T *data, size_t count, size_t elementsPerRow)
{
    if (elementsPerRow == 0) {
        ERROR_LOG("elementsPerRow cannot be zero");
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(10) << data[i];
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void DoPrintFp16Data(const aclFloat16 *data, size_t count, size_t elementsPerRow)
{
    if (elementsPerRow == 0) {
        ERROR_LOG("elementsPerRow cannot be zero");
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(10) << std::setprecision(4) << aclFloat16ToFloat(data[i]);
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

template<typename T>
void PrintDataImpl(const void *data, size_t count, size_t elementsPerRow)
{
    const T *typedData = static_cast<const T*>(data);
    DoPrintData(typedData, count, elementsPerRow);
}

void PrintData(const void *data, size_t count, aclDataType dataType, size_t elementsPerRow)
{
    if (data == nullptr) {
        ERROR_LOG("Print data failed. data is nullptr");
        return;
    }

    switch (dataType) {
        case ACL_BOOL:
            PrintDataImpl<bool>(data, count, elementsPerRow);
            break;
        case ACL_INT8:
            PrintDataImpl<int8_t>(data, count, elementsPerRow);
            break;
        case ACL_UINT8:
            PrintDataImpl<uint8_t>(data, count, elementsPerRow);
            break;
        case ACL_INT16:
            PrintDataImpl<int16_t>(data, count, elementsPerRow);
            break;
        case ACL_UINT16:
            PrintDataImpl<uint16_t>(data, count, elementsPerRow);
            break;
        case ACL_INT32:
            PrintDataImpl<int32_t>(data, count, elementsPerRow);
            break;
        case ACL_UINT32:
            PrintDataImpl<uint32_t>(data, count, elementsPerRow);
            break;
        case ACL_INT64:
            PrintDataImpl<int64_t>(data, count, elementsPerRow);
            break;
        case ACL_UINT64:
            PrintDataImpl<uint64_t>(data, count, elementsPerRow);
            break;
        case ACL_FLOAT16:
            DoPrintFp16Data(static_cast<const aclFloat16 *>(data), count, elementsPerRow);
            break;
        case ACL_FLOAT:
            PrintDataImpl<float>(data, count, elementsPerRow);
            break;
        case ACL_DOUBLE:
            PrintDataImpl<double>(data, count, elementsPerRow);
            break;
        default:
            ERROR_LOG("Unsupported type: %d", dataType);
    }
}

void OpRunner::PrintInput(size_t index, size_t numElementsPerRow)
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numInputs_);
        return;
    }

    auto desc = opDesc_->inputDesc[index];
    PrintData(hostInputs_[index], GetInputElementCount(index), aclGetTensorDescType(desc), numElementsPerRow);
}

void OpRunner::PrintOutput(size_t index, size_t numElementsPerRow)
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return;
    }

    auto desc = opDesc_->outputDesc[index];
    PrintData(hostOutputs_[index], GetOutputElementCount(index), aclGetTensorDescType(desc), numElementsPerRow);
}
// Original main.cpp content
namespace {
constexpr int32_t INPUT_BUFFER_BIAS = 2; 

OperatorDesc CreateOpDesc()
{
    std::vector<int64_t> shapeA { RANK_M, RANK_K };
    std::vector<int64_t> shapeB { RANK_K, RANK_N };
    std::vector<int64_t> shapeBias {};
    std::vector<int64_t> shapeC { RANK_M / RANK_DIM, RANK_N };
    aclDataType dataTypeA = ACL_FLOAT16;
    aclDataType dataTypeB = ACL_FLOAT16;
    aclDataType dataTypeBias = ACL_FLOAT16;
    aclDataType dataTypeC = ACL_FLOAT16;
    aclFormat format = ACL_FORMAT_ND;

    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(dataTypeA, shapeA.size(), shapeA.data(), format);
    opDesc.AddInputTensorDesc(dataTypeB, shapeB.size(), shapeB.data(), format);
    opDesc.AddInputTensorDesc(dataTypeBias, shapeBias.size(), shapeBias.data(), format);
    opDesc.AddOutputTensorDesc(dataTypeC, shapeC.size(), shapeC.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner, uint32_t rankId)
{
    size_t fileSize = 0;
    ReadFile("../input/input_x1_" + std::to_string(rankId) + ".bin", fileSize,
        runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("../input/input_x2_" + std::to_string(rankId) + ".bin", fileSize,
        runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    ReadFile("../input/input_bias_" + std::to_string(rankId) + ".bin", fileSize,
        runner.GetInputBuffer<void>(INPUT_BUFFER_BIAS), runner.GetInputSize(INPUT_BUFFER_BIAS));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner, uint32_t rankId)
{
    WriteFile("../output/out_" + std::to_string(rankId) + ".bin", runner.GetOutputBuffer<void>(0),
        runner.GetOutputSize(0));
    INFO_LOG("Write output success");
    return true;
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        constexpr mode_t OUTPUT_DIR_PERMISSIONS = 0700;
        if (mkdir(output.c_str(), OUTPUT_DIR_PERMISSIONS) != 0) {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    if (aclInit(NULL) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    for (int32_t i = 0; i < RANK_DIM; i++) {
        if (aclrtSetDevice(i) != ACL_SUCCESS) {
            ERROR_LOG("Set device failed. deviceId is %u", i);
            for (uint32_t j = 0; j < i; j++) {
                (void)aclrtResetDevice(j);
            }
            (void)aclFinalize();
            return false;
        }
    }
    return true;
}

bool RunOp(uint32_t rankId, HcclComm &comm)
{
    aclrtContext context;
    if (aclrtCreateContext(&context, rankId) != ACL_SUCCESS) {
        ERROR_LOG("Create context failed. deviceId is %u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    aclrtStream stream;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        ERROR_LOG("Create stream failed. deviceId is %u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    if (aclrtSetCurrentContext(context) != ACL_SUCCESS) {
        ERROR_LOG("Set current context failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    char group[128] = {0};
    if (HcclGetCommName(comm, group) != HCCL_SUCCESS) {
        ERROR_LOG("Hccl get comm name failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    OperatorDesc opDesc = CreateOpDesc();
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    if (!SetInputData(opRunner, rankId)) {
        ERROR_LOG("Set input data failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    if (!opRunner.RunOp(group, stream)) {
        ERROR_LOG("Run op failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    if (!ProcessOutputData(opRunner, rankId)) {
        ERROR_LOG("Process output data failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    (void)HcclCommDestroy(comm);
    (void)aclrtDestroyStream(stream);
    (void)aclrtDestroyContext(context);
    (void)aclrtResetDevice(rankId);

    INFO_LOG("Run op success, deviceId=%u", rankId);
    return true;
}
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }

    INFO_LOG("Init resource success");

    HcclComm comms[RANK_DIM];
    int32_t devices[RANK_DIM];
    for (int32_t i = 0; i < RANK_DIM; i++) {
        devices[i] = i;
    }
    if (HcclCommInitAll(RANK_DIM, devices, comms) != HCCL_SUCCESS) {
        ERROR_LOG("Hccl comm init failed.");
        (void)aclFinalize();
        return FAILED;
    }

    // run with multithread
    std::vector<std::unique_ptr<std::thread>> threads(RANK_DIM);
    for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
        threads[rankId].reset(new(std::nothrow) std::thread(&RunOp, rankId, std::ref(comms[rankId])));
    }
    for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
        threads[rankId]->join();
    }

    (void)aclFinalize();
    return SUCCESS;
}
