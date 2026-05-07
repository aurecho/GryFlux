#include "context/acl_infer_context.h"

#include "context/acl_environment.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace resnet {
namespace {

void CheckAcl(aclError error, const char* action) {
    if (error == ACL_SUCCESS) {
        return;
    }

    throw std::runtime_error(
        std::string(action) + " failed, aclError=" + std::to_string(error));
}

void SetError(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

template <typename T>
T* CheckNotNull(T* pointer, const char* action) {
    if (pointer != nullptr) {
        return pointer;
    }

    throw std::runtime_error(std::string(action) + " returned null");
}

void DestroyDataset(aclmdlDataset*& dataset) noexcept {
    if (dataset == nullptr) {
        return;
    }

    const size_t buffer_count = aclmdlGetDatasetNumBuffers(dataset);
    for (size_t index = 0; index < buffer_count; ++index) {
        aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(dataset, index);
        if (data_buffer != nullptr) {
            aclDestroyDataBuffer(data_buffer);
        }
    }
    aclmdlDestroyDataset(dataset);
    dataset = nullptr;
}

}  // namespace

AclInferContext::AclInferContext(std::string model_path, int device_id)
    : model_path_(std::move(model_path)),
      device_id_(device_id) {}

AclInferContext::~AclInferContext() {
    cleanup();
}

bool AclInferContext::init(std::string* error) {
    if (initialized_) {
        return true;
    }

    if (!AclEnvironment::acquire(device_id_, error)) {
        return false;
    }
    acl_ready_ = true;

    try {
        setDevice();
        loadModel();
        createInputBuffers();
        createOutputBuffers();
        initialized_ = true;
    } catch (const std::exception& exception) {
        SetError(error, exception.what());
        cleanup();
        return false;
    }

    return true;
}

size_t AclInferContext::getInputBufferSize() const {
    if (input_buffers_.empty()) {
        return 0;
    }
    return input_buffers_.front().size;
}

void AclInferContext::run(const void* input_data, size_t input_size) {
    if (!initialized_) {
        throw std::runtime_error("AclInferContext is not initialized");
    }
    if (input_buffers_.empty()) {
        throw std::runtime_error("Model input count is zero");
    }

    ModelInput& input = input_buffers_.front();
    if (input_size != input.size) {
        throw std::runtime_error("Input buffer size mismatch");
    }

    setDevice();
    CheckAcl(aclrtMemcpy(input.device_buffer,
                         input.size,
                         input_data,
                         input_size,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(host_to_device)");
    CheckAcl(aclmdlExecute(model_id_, input_dataset_, output_dataset_),
             "aclmdlExecute");

    for (auto& output : output_buffers_) {
        CheckAcl(aclrtMemcpy(output.host_buffer,
                             output.size,
                             output.device_buffer,
                             output.size,
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy(device_to_host)");
    }
}

const void* AclInferContext::getOutputHostBuffer(size_t index) const {
    return outputBuffer(index).host_buffer;
}

size_t AclInferContext::getOutputSize(size_t index) const {
    return outputBuffer(index).size;
}

void AclInferContext::setDevice() const {
    CheckAcl(aclrtSetDevice(device_id_), "aclrtSetDevice");
}

void AclInferContext::loadModel() {
    CheckAcl(aclmdlLoadFromFile(model_path_.c_str(), &model_id_),
             "aclmdlLoadFromFile");
    model_loaded_ = true;

    model_desc_ = CheckNotNull(aclmdlCreateDesc(), "aclmdlCreateDesc");
    CheckAcl(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc");
}

void AclInferContext::createInputBuffers() {
    input_dataset_ = CheckNotNull(aclmdlCreateDataset(), "aclmdlCreateDataset(input)");

    const size_t input_count = aclmdlGetNumInputs(model_desc_);
    input_buffers_.reserve(input_count);
    for (size_t index = 0; index < input_count; ++index) {
        ModelInput input;
        input.size = aclmdlGetInputSizeByIndex(model_desc_, index);
        CheckAcl(aclrtMalloc(&input.device_buffer, input.size, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(input)");

        aclDataBuffer* buffer = CheckNotNull(
            aclCreateDataBuffer(input.device_buffer, input.size),
            "aclCreateDataBuffer(input)");
        CheckAcl(aclmdlAddDatasetBuffer(input_dataset_, buffer),
                 "aclmdlAddDatasetBuffer(input)");
        input_buffers_.push_back(input);
    }
}

void AclInferContext::createOutputBuffers() {
    output_dataset_ = CheckNotNull(aclmdlCreateDataset(), "aclmdlCreateDataset(output)");

    const size_t output_count = aclmdlGetNumOutputs(model_desc_);
    output_buffers_.reserve(output_count);
    for (size_t index = 0; index < output_count; ++index) {
        ModelOutput output;
        output.size = aclmdlGetOutputSizeByIndex(model_desc_, index);

        CheckAcl(aclrtMalloc(&output.device_buffer, output.size, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(output)");
        CheckAcl(aclrtMallocHost(&output.host_buffer, output.size),
                 "aclrtMallocHost(output)");

        aclDataBuffer* buffer = CheckNotNull(
            aclCreateDataBuffer(output.device_buffer, output.size),
            "aclCreateDataBuffer(output)");
        CheckAcl(aclmdlAddDatasetBuffer(output_dataset_, buffer),
                 "aclmdlAddDatasetBuffer(output)");
        output_buffers_.push_back(output);
    }
}

const AclInferContext::ModelOutput& AclInferContext::outputBuffer(size_t index) const {
    if (index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }
    return output_buffers_[index];
}

void AclInferContext::cleanup() noexcept {
    if (acl_ready_) {
        aclrtSetDevice(device_id_);
    }

    destroyDatasets();
    destroyBuffers();
    unloadModel();
    initialized_ = false;

    if (acl_ready_) {
        AclEnvironment::release(device_id_);
        acl_ready_ = false;
    }
}

aclmdlIODims AclInferContext::getInputDims(size_t input_index) const {
    if (input_index >= input_buffers_.size()) {
        throw std::runtime_error("Input index is out of range");
    }

    aclmdlIODims dims{};
    CheckAcl(aclmdlGetInputDims(model_desc_, input_index, &dims), "aclmdlGetInputDims");
    return dims;
}

aclmdlIODims AclInferContext::getOutputDims(size_t output_index) const {
    if (output_index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }

    aclmdlIODims dims{};
    CheckAcl(aclmdlGetOutputDims(model_desc_, output_index, &dims), "aclmdlGetOutputDims");
    return dims;
}

aclmdlIODims AclInferContext::getCurrentOutputDims(size_t output_index) const {
    if (output_index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }

    aclmdlIODims dims{};
    CheckAcl(
        aclmdlGetCurOutputDims(model_desc_, output_index, &dims),
        "aclmdlGetCurOutputDims");
    return dims;
}

aclFormat AclInferContext::getInputFormat(size_t input_index) const {
    if (input_index >= input_buffers_.size()) {
        throw std::runtime_error("Input index is out of range");
    }
    return aclmdlGetInputFormat(model_desc_, input_index);
}

aclFormat AclInferContext::getOutputFormat(size_t output_index) const {
    if (output_index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }
    return aclmdlGetOutputFormat(model_desc_, output_index);
}

aclDataType AclInferContext::getInputDataType(size_t input_index) const {
    if (input_index >= input_buffers_.size()) {
        throw std::runtime_error("Input index is out of range");
    }
    return aclmdlGetInputDataType(model_desc_, input_index);
}

aclDataType AclInferContext::getOutputDataType(size_t output_index) const {
    if (output_index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }
    return aclmdlGetOutputDataType(model_desc_, output_index);
}

void AclInferContext::unloadModel() noexcept {
    if (model_desc_ != nullptr) {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }

    if (model_loaded_) {
        aclmdlUnload(model_id_);
        model_id_ = 0;
        model_loaded_ = false;
    }
}

void AclInferContext::destroyDatasets() noexcept {
    DestroyDataset(input_dataset_);
    DestroyDataset(output_dataset_);
}

void AclInferContext::destroyBuffers() noexcept {
    for (auto& input : input_buffers_) {
        if (input.device_buffer != nullptr) {
            aclrtFree(input.device_buffer);
            input.device_buffer = nullptr;
        }
    }
    input_buffers_.clear();

    for (auto& output : output_buffers_) {
        if (output.device_buffer != nullptr) {
            aclrtFree(output.device_buffer);
            output.device_buffer = nullptr;
        }
        if (output.host_buffer != nullptr) {
            aclrtFreeHost(output.host_buffer);
            output.host_buffer = nullptr;
        }
    }
    output_buffers_.clear();
}

std::vector<std::shared_ptr<GryFlux::Context>> CreateAclInferContexts(
    const std::string& om_model_path,
    int device_id,
    size_t instance_count) {
    if (instance_count == 0) {
        throw std::runtime_error("ACL context instance count must be greater than zero");
    }

    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
    contexts.reserve(instance_count);
    for (size_t index = 0; index < instance_count; ++index) {
        auto context = std::make_shared<AclInferContext>(om_model_path, device_id);
        std::string init_error;
        if (!context->init(&init_error)) {
            throw std::runtime_error("AclInferContext init failed: " + init_error);
        }
        contexts.push_back(std::move(context));
    }
    return contexts;
}

}  // namespace resnet
