# GryFlux Framework - resnet_Atlas

## 示例说明

本示例用于演示：

- 如何在 GryFlux 中搭建基于 Atlas NPU 的图像分类异步流水线
- 如何通过 `DataSource -> DAG -> DataConsumer` 组织批量评估流程
- 如何把预处理、推理、后处理拆分为独立节点并行调度
- 如何使用 `ResourcePool` 注册 Atlas NPU 资源并自动分发推理任务
- 如何在全部样本完成后统一统计 Top-1 / Top-5 / FPS

本应用入口在 `src/app/resnet_Atlas/resnet_Atlas.cpp`，可执行文件名为 `resnet_Atlas`。

## 快速上手

下面参照 `example/README.md` 的方式，说明当前 `resnet_Atlas` 是如何定义数据包、节点、资源上下文、DAG 和异步管道的。

### 1) 定义数据包（DataPacket）

数据包是流经整个 DAG 的载体。  
在本示例中，`ResNetPacket` 负责保存：

- 图片路径
- ground truth 标签
- 预处理后的输入 tensor
- 推理输出 logits
- Top-1 / Top-5 结果

```cpp
struct ResNetPacket : public GryFlux::DataPacket {
    uint64_t packet_id = 0;

    std::string image_path;
    int ground_truth_label = -1;
    std::vector<float> preprocessed_data;
    std::vector<float> logits;
    int top1_class = -1;
    bool top5_correct = false;

    ResNetPacket()
        : preprocessed_data(3 * 224 * 224),
          logits(1000)
    {}

    uint64_t getIdx() const override {
        return packet_id;
    }
};
```

这里在构造函数中预分配了输入和输出缓存，避免运行时频繁分配内存。

### 2) 定义节点（NodeBase）

节点只需要继承 `GryFlux::NodeBase` 并实现 `execute()`。

- `InputNode`
- `PreprocessNode`
- `InferNode`
- `PostprocessNode`
- `OutputNode`

其中 `InputNode` 和 `OutputNode` 是空透传节点，用来让图结构和 `example/example.cpp` 保持一致。

#### 2.1 `PreprocessNode`

`PreprocessNode` 负责将图片转换为模型输入格式，主要步骤包括：

- 读取图像
- 按短边缩放到 `256`
- 中心裁剪到 `224x224`
- BGR 转 RGB
- 转成 NCHW 排布
- 用 ImageNet 均值方差归一化

```cpp
cv::Mat image = cv::imread(p.image_path, cv::IMREAD_COLOR);
cv::resize(image, resized_image, cv::Size(new_w, new_h));
cv::Rect roi(x, y, IMG_WIDTH, IMG_HEIGHT);
cv::Mat cropped_image = resized_image(roi).clone();
```

归一化参数为：

```cpp
const float MEAN_RGB[3] = {0.485f, 0.456f, 0.406f};
const float STD_RGB[3]  = {0.229f, 0.224f, 0.225f};
```

#### 2.2 `InferNode`

`InferNode` 的职责很简单，就是把 packet 中的输入 tensor 交给 `AclInferContext` 执行推理：

```cpp
void InferNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& resnet_packet = static_cast<ResNetPacket&>(packet);
    auto& infer_context = static_cast<resnet::AclInferContext&>(ctx);

    const size_t input_bytes =
        resnet_packet.preprocessed_data.size() * sizeof(float);
    infer_context.run(resnet_packet.preprocessed_data.data(), input_bytes);
}
```

#### 2.3 `PostprocessNode`

`PostprocessNode` 负责从 `logits` 中提取 Top-5 并判断分类结果是否命中 ground truth：

```cpp
std::partial_sort(
    score_index_pairs.begin(),
    score_index_pairs.begin() + 5,
    score_index_pairs.end(),
    std::greater<std::pair<float, int>>()
);

p.top1_class = score_index_pairs[0].second;
```

最终会得到：

- `top1_class`
- `top5_correct`

### 3) 定义 Context（资源上下文）

当某个节点需要“受限硬件资源”时，需要定义一个 `GryFlux::Context` 子类。  
本示例中的资源上下文是 `AclInferContext`。

它整体参考了 `ct/OmModelRunner` 的职责划分，但保留的是更贴合当前分类示例的简化版。当前封装了：

- ACL 全局生命周期接入
- OM 模型加载与 `model_desc`
- 输入输出 device buffer
- 输入输出 dataset / data buffer
- 单次 `run()` 推理路径

初始化阶段会完成：

- `AclEnvironment::acquire()`
- `aclrtSetDevice`
- `aclmdlLoadFromFile`
- `aclmdlGetDesc`
- 输入输出显存分配
- dataset / data buffer 创建

推理接口现在收敛为：

```cpp
void run(const void* input_data, size_t input_size);
```

调用 `run()` 时会在内部完成：

- Host -> Device 拷贝
- `aclmdlExecute`
- Device -> Host 拷贝

### 4) 注册资源池（ResourcePool）

当节点需要 Atlas NPU 资源时，先注册对应资源类型。

当前示例注册了同一张设备上的两个推理上下文：

```cpp
auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
resource_pool->registerResourceType(
    "npu",
    resnet::CreateAclInferContexts(
        options.model_path,
        0,
        kNpuInstanceCount));
```

这意味着 `inference` 节点在运行时会从 `npu` 资源池中获取一个可用上下文。当前主程序默认只注册 `device 0` 上的 2 个实例，不是双卡轮转。

### 5) 构建 DAG（GraphTemplate + TemplateBuilder）

通过 `GraphTemplate::buildOnce()` 定义拓扑：

- `setInputNode<T>`：输入节点
- `addTask<T>`：中间任务节点
- `setOutputNode<T>`：输出节点

当前图结构为：

```cpp
auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
    [](GryFlux::TemplateBuilder *builder) {
        builder->setInputNode<InputNode>("input");
        builder->addTask<PreprocessNode>("preprocess", "", {"input"});
        builder->addTask<PipelineNodes::InferNode>("inference", "npu", {"preprocess"});
        builder->addTask<PostprocessNode>("postprocess", "", {"inference"});
        builder->setOutputNode<OutputNode>("output", {"postprocess"});
    }
);
```

也就是说，本示例的依赖关系是：

`input -> preprocess -> inference -> postprocess -> output`

### 6) 运行异步管道（AsyncPipeline）

```cpp
auto source = std::make_shared<ResNetDataSource>(datasetDir, gt_map);
auto consumer = std::make_shared<ResNetResultConsumer>(gt_map.size());

GryFlux::AsyncPipeline pipeline(
    source,
    graphTemplate,
    resourcePool,
    consumer,
    kThreadPoolSize,
    kMaxActivePackets
);
```

当前主程序直接阻塞调用 `pipeline.run()`：

```cpp
pipeline.run();
```

如果编译时开启了 profiling，还会在运行前启用 profiler，并在结束后输出统计和导出时间线：

```cpp
if constexpr (GryFlux::Profiling::kBuildProfiling) {
    pipeline.setProfilingEnabled(true);
}

pipeline.run();

if constexpr (GryFlux::Profiling::kBuildProfiling) {
    pipeline.printProfilingStats();
    pipeline.dumpProfilingTimeline("graph_timeline_resnet.json");
}
```

## 示例 DAG 结构

DAG 结构图：

![Computation Graph](assets/chart.svg)

当前模块对应关系：

- `source`：`ResNetDataSource`
- `packet`：`ResNetPacket`
- `nodes`：`InputNode -> PreprocessNode -> InferNode -> PostprocessNode -> OutputNode`
- `context`：`AclInferContext`
- `consumer`：`ResNetResultConsumer`

## 资源绑定

当前资源绑定如下：

- `CPU(绿)`：`PreprocessNode`、`PostprocessNode`
- `Atlas NPU(蓝)`：`InferNode`

其中 `AclInferContext` 当前注册了两个实例：

- `Device 0`

这意味着：

- 预处理 / 后处理由 CPU 线程池并发执行
- 推理阶段由同一张 Atlas 设备上的两个上下文并行承载

## 预处理与后处理逻辑

和 `example` 中的“人为 sleep 模拟耗时”不同，`resnet_Atlas` 的时间主要消耗在真实工作上：

- CPU 图像解码、缩放、裁剪、归一化
- NPU 推理执行
- Top-K 后处理

这里没有显式的 `delayMs` 配置，但你仍然可以把它看成一个标准的“CPU + 受限硬件资源 + CPU” 三段式流水线。

## 管道运行参数

当前主程序中定义了两个关键参数：

```cpp
constexpr size_t kThreadPoolSize = 8;
constexpr size_t kMaxActivePackets = 16;
```

它们分别控制：

- `kThreadPoolSize`：CPU 节点并发度
- `kMaxActivePackets`：系统中同时在途的数据包数量上限

如果要分析吞吐，通常需要一起考虑：

- CPU 预处理速度
- 单卡多上下文 NPU 推理吞吐
- `kMaxActivePackets` 是否足够覆盖系统在途深度

## 构建与运行

### 1) 构建

当前仓库顶层 `build.sh` 只会构建 `src/app/example`。  
如果要构建 `resnet_Atlas`，建议单独构建这个子目录：

```bash
cmake -S src/app/resnet_Atlas -B build/resnet_Atlas
cmake --build build/resnet_Atlas -j"$(nproc)"
```

可执行文件位于：

```bash
build/resnet_Atlas/resnet_Atlas
```

### 2) 运行

```bash
/root/workspace/zjx/GryFlux/build/resnet_Atlas/resnet_Atlas -m <om_model_path> -d <dataset_dir> -g <gt_file_path>
```

程序启动参数共 3 个：

- `om_model_path`：Atlas OM 模型路径
- `dataset_dir`：图片目录路径
- `gt_file_path`：标签文件路径

其中 `dataset_dir` 需要和 `gt_file_path` 中记录的相对路径对齐。  
例如标签文件里如果是 `n03445777/xxx.JPEG`，而图片实际在 `val/n03445777/xxx.JPEG` 下，那么这里应传 `.../val`，不是数据集根目录。

程序会输出：

- 已处理数量
- 总耗时
- FPS
- Top-1 / Top-5 准确率

## Profiling 与时间线

如果编译时启用了 profiling：

```bash
cmake -S src/app/resnet_Atlas -B build/resnet_Atlas_profile -DCMAKE_CXX_FLAGS=-DGRYFLUX_BUILD_PROFILING=1
cmake --build build/resnet_Atlas_profile -j"$(nproc)"
```

程序运行结束后会额外输出 profiling 统计，并导出：

- `graph_timeline_resnet.json`

导出位置是程序启动时的当前工作目录。

`graph_timeline_resnet.json` 在导出前会按事件时间戳排序，避免多线程记录时出现少量乱序，便于 timeline viewer 正确绘图。

仓库中的 `assets/timeline_resnet.json` 是一份真实 profiling 样例，生成方式为：

- 同一张真实图片重复 12 次
- 先做 1 轮 warm-up
- 再做 1 轮正式录制

这份样例仍然对应真实的 5 节点执行链路：

- `input`
- `preprocess`
- `inference`
- `postprocess`
- `output`

之所以采用这份样例，而不是直接使用杂图混跑结果，是因为它更容易观察：

- CPU `preprocess` 与双 `npu` context 的并发关系
- `input / preprocess / inference / postprocess / output` 的真实先后顺序
- warm-up 后更稳定的耗时分布

你可以像 `example/README.md` 一样，用网页查看时间线：

```text
http://profile.grifcc.top:8076/
```

操作方式：

1. 浏览器打开该页面
2. 选择 `graph_timeline_resnet.json`
3. 生成 packet 级 timeline

同时目录中还提供了：

- `assets/chart.svg`：简化 DAG 图

## 指标说明

`ResNetResultConsumer` 在最后输出：

- `总耗时`
- `吞吐量 (FPS)`
- `Top-1 准确率`
- `Top-5 准确率`

实现方式是：

- 每处理完一个 packet 就更新统计值
- `pipeline.run()` 返回后调用 `printMetrics()` 打印汇总结果

## 退出与稳定性说明

为保证“处理完成后正常退出”，当前实现采用：

- GT 文件检查放在资源初始化之前
- 主线程直接调用 `pipeline.run()`
- `DataSource` 在耗尽时正确更新 `hasMore`
- 所有 ACL 资源最终通过 `AclEnvironment` 统一释放

如果出现无法退出或提前退出，优先检查：

- 标签文件是否为空
- 数据目录与标签文件是否一一对应
- 模型路径是否可读
- Ascend 运行时环境变量是否正确
- OpenCV 是否能正常读取输入图片

## 目录结构

- `resnet_Atlas.cpp`: 主程序入口
- `source/resnet_data_source.h`: 数据源
- `packet/resnet_packet.h`: 数据包定义
- `context/acl_environment.h/.cpp`: ACL 生命周期管理
- `context/acl_infer_context.h/.cpp`: Atlas 推理上下文
- `nodes/Input/InputNode.cpp`: 输入节点
- `nodes/Preprocess/PreprocessNode.cpp`: 图像预处理
- `nodes/Infer/InferNode.cpp`: NPU 推理执行
- `nodes/Postprocess/PostprocessNode.cpp`: Top-K 后处理
- `nodes/Output/OutputNode.cpp`: 输出节点
- `consumer/resnet_result_consumer.h`: 结果统计
- `assets/chart.svg`: DAG 图
