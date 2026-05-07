#pragma once

#include "framework/context.h"

#include "acl/acl.h"
#include "acl/acl_mdl.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace resnet {

class AclInferContext : public GryFlux::Context {
public:
    AclInferContext(std::string model_path, int device_id);
    ~AclInferContext() override;

    bool init(std::string* error);

    int getDeviceId() const { return device_id_; }

    size_t getInputBufferSize() const;
    void run(const void* input_data, size_t input_size);

    size_t getNumOutputs() const { return output_buffers_.size(); }
    const void* getOutputHostBuffer(size_t index) const;
    size_t getOutputSize(size_t index) const;
    aclmdlIODims getInputDims(size_t input_index) const;
    aclmdlIODims getOutputDims(size_t output_index) const;
    aclmdlIODims getCurrentOutputDims(size_t output_index) const;
    aclFormat getInputFormat(size_t input_index) const;
    aclFormat getOutputFormat(size_t output_index) const;
    aclDataType getInputDataType(size_t input_index) const;
    aclDataType getOutputDataType(size_t output_index) const;

private:
    struct ModelInput {
        void* device_buffer = nullptr;
        size_t size = 0;
    };

    struct ModelOutput {
        void* device_buffer = nullptr;
        void* host_buffer = nullptr;
        size_t size = 0;
    };

    void setDevice() const;
    void loadModel();
    void createInputBuffers();
    void createOutputBuffers();
    const ModelOutput& outputBuffer(size_t index) const;
    void cleanup() noexcept;
    void destroyDatasets() noexcept;
    void destroyBuffers() noexcept;
    void unloadModel() noexcept;

    std::string model_path_;
    bool acl_ready_ = false;
    bool initialized_ = false;
    bool model_loaded_ = false;
    int device_id_ = 0;

    uint32_t model_id_ = 0;
    aclmdlDesc* model_desc_ = nullptr;

    std::vector<ModelInput> input_buffers_;
    aclmdlDataset* input_dataset_ = nullptr;

    std::vector<ModelOutput> output_buffers_;
    aclmdlDataset* output_dataset_ = nullptr;
};

std::vector<std::shared_ptr<GryFlux::Context>> CreateAclInferContexts(
    const std::string& om_model_path,
    int device_id,
    size_t instance_count);

}  // namespace resnet
