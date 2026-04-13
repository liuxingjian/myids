# AI-IDS: 基于AI的实时入侵检测系统

一个基于深度学习和RKNN加速的实时网络流量入侵检测系统，能够实时监控网络流量并识别潜在的异常行为和入侵攻击。

## 📋 目录

- [项目简介](#项目简介)
- [主要特性](#主要特性)
- [系统要求](#系统要求)
- [项目结构](#项目结构)
- [安装与配置](#安装与配置)
- [使用方法](#使用方法)
  - [流量采集](#1-流量采集)
  - [数据预处理](#2-数据预处理)
  - [实时检测](#3-实时检测)
  - [训练与跨机模型转换](#4-训练与跨机模型转换)
- [配置说明](#配置说明)
- [注意事项](#注意事项)

## 🎯 项目简介

本项目实现了一个端到端的实时入侵检测系统，通过以下步骤实现网络流量的实时监控和异常检测：

1. **流量采集**：使用 NFStream 库实时捕获网络数据包并提取流量特征
2. **数据预处理**：对原始流量特征进行标准化、归一化和编码处理
3. **模型推理**：使用 RKNN 加速的深度学习模型进行实时异常检测

系统采用滑动窗口机制，能够基于多条流的上下文信息进行更准确的异常检测。

## ✨ 主要特性

- 🔄 **实时流量监控**：基于 NFStream 的实时网络流量采集和分析
- 🤖 **AI 驱动检测**：使用 Transformer 等深度学习模型进行异常检测
- ⚡ **RKNN 加速**：利用 Rockchip NPU 进行模型推理加速
- 📊 **滑动窗口机制**：基于多流上下文信息提升检测准确性
- 🔧 **灵活配置**：支持自定义特征提取和预处理参数
- 📈 **详细日志**：提供完整的推理结果和异常告警信息

## 💻 系统要求

- **操作系统**：Linux (推荐 Ubuntu 18.04+)
- **Python**：3.7+
- **硬件**：支持 RKNN 的 Rockchip 设备（如 RK3588）
- **依赖库**：
  - `nfstream`
  - `pandas`
  - `numpy`
  - `rknnlite`
  - `scikit-learn`

## 📁 项目结构

```
ai-ids/
├── flow_collection/          # 流量采集模块
│   ├── flow_collection_to_csv.py    # 将流量特征保存为CSV
│   └── flow_collection_to_pd.py     # 将流量特征输出为DataFrame
├── preprocessing/            # 数据预处理模块
│   ├── extract_preprocessor_config.py  # 提取预处理配置
│   ├── preprocess_dataframe.py         # 数据预处理工具类
│   └── config.json                    # 预处理配置文件
├── models/                   # 模型文件目录
│   ├── transformer_multi_class_model.rknn
│   ├── transformer_multi_class_model.meta.json
│   └── t_multi_class.onnx
├── config/                   # 配置文件目录
│   └── config.json          # 主配置文件
├── tools/
│   └── onnx_to_rknn.py      # ONNX 转 RKNN 独立脚本
├── main.py                   # 主程序入口
└── README.md                # 项目说明文档
```

## 🚀 安装与配置

### 1. 安装依赖

```bash
pip install nfstream pandas numpy scikit-learn
```

### 2. 安装 RKNN Lite

根据您的硬件平台安装对应的 RKNN Lite 库。参考 Rockchip 官方文档进行安装。

### 3. 准备模型文件

将训练好的模型转换为 RKNN 格式，并放置在 `models/` 目录下。

## 📖 使用方法

### 1. 流量采集

#### 方式一：保存为 CSV 文件

将网络接口 `eth0` 的流量特征提取并保存到 CSV 文件：

```bash
python flow_collection/flow_collection_to_csv.py
```

输出文件：`./flows.csv`


#### 转成标准格式


python tools/raw_to_nfv1.py --input ./flows.0.csv --attack attack --output ./cache/converted_attack.csv


#### 方式二：实时输出到终端

将流量特征实时打印到终端，可用于云平台传输或直接推理：

```bash
python flow_collection/flow_collection_to_pd.py
```

### 2. 数据预处理

#### 2.1 提取预处理配置

从训练数据集提取预处理参数（归一化参数、独热编码映射等）并保存为 JSON 配置文件：

```bash
python preprocessing/extract_preprocessor_config.py \
    --dataset /path/to/dataset.csv \
    --output config/config.json \
    --numerical-features IN_PKTS OUT_PKTS IN_BYTES OUT_BYTES FLOW_DURATION_MILLISECONDS \
    --categorical-features L4_SRC_PORT L4_DST_PORT PROTOCOL L7_PROTO TCP_FLAGS \
    --top-k 32 \
    --window-size 8
```

**参数说明**：
- `--dataset`: 训练数据集的 CSV 文件路径
- `--output`: 输出的配置文件路径
- `--numerical-features`: 数值型特征列表（将进行归一化）
- `--categorical-features`: 类别型特征列表（将进行 Top-K 独热编码）
- `--top-k`: 类别型特征的 Top-K 编码数量
- `--window-size`: 滑动窗口大小（用于实时推理）

#### 2.2 使用预处理器

在代码中使用预处理器对数据进行预处理：

```python
from preprocessing.preprocess_dataframe import DataFramePreprocessor

# 加载预处理器
preprocessor = DataFramePreprocessor.load_from_json('config/config.json')

# 预处理 DataFrame
df = pd.read_csv('data.csv')
preprocessed_df = preprocessor.preprocess_dataframe(df)
```

**预处理功能**：
- ✅ 自动去除源/目标 IP 地址字段
- ✅ 数值型特征归一化（Min-Max 归一化）
- ✅ 类别型特征 Top-K 独热编码
- ✅ 支持滑动窗口数据组织

### 3. 实时检测

运行主程序进行实时流量监控和异常检测（`-m` 为必填）：

```bash
python main.py -m ./models/transformer_multi_class_model.rknn [网络接口]
```

**示例**：
```bash
# 监控 eth0 接口
python main.py -m ./models/transformer_multi_class_model.rknn --auto-align-config eth0

# 监控其他接口
python main.py -m ./models/transformer_multi_class_model.rknn --auto-align-config enp1s0

# 显式指定模型元信息（推荐）
python main.py -m ./models/transformer_multi_class_model.rknn \
  --model-meta ./models/transformer_multi_class_model.meta.json \
  --auto-align-config

# 若元信息中暂无标签，可临时手工覆盖标签名
python main.py -m ./models/transformer_multi_class_model.rknn \
  --class-labels Benign,tcp,udp --benign-label Benign --auto-align-config
```

**程序流程**：
1. 加载 RKNN 模型和预处理配置
2. 初始化 NFStream 流收集器
3. 实时捕获网络数据包并提取流量特征
4. 使用滑动窗口累积流量数据
5. 当窗口满时，进行数据预处理和模型推理
6. 输出推理结果，检测到异常时触发告警

**输出示例**：
```
[Flow #100] Inference completed
  Flow info: {'src_port': 443, 'dst_port': 54321, 'protocol': 6, 'l7_proto': 'TLS'}
  Inference time: 15.23 ms
  Output shape: (1, 3)
  Max probability: 0.9766
  Task type: multiclass
  Predicted label: Benign, confidence: 0.9766 (class_index=0)

⚠️  ALERT: Attack detected (tcp, probability: 0.8234)
  🎯 Current Flow (most recent):
      Source IP: 192.168.1.100
      Destination IP: 10.0.0.50
      Source Port: 54321
      Destination Port: 80
      Protocol: 6 (HTTP)
```

### 4. 训练与跨机模型转换

推荐使用“训练机导出 ONNX + metadata，转换机转 RKNN”的流程：

1. 在训练机运行 `FlowTransformer_MultiClassification_Extension/multi_classification_demo.ipynb`。
2. 导出 cell 会生成：
  - `models/t_multi_class.onnx`
  - `models/t_multi_class.meta.json`
3. `*.meta.json` 会包含：
  - 模型输入顺序：`model_input_names`
  - 类别特征维度：`model_expected_dims`
  - 类别映射：`class_map`、`class_labels`
  - 良性类别信息：`benign_label`、`benign_class_index`
4. 将 ONNX + metadata 拷贝到转换机，执行：

```bash
python tools/onnx_to_rknn.py \
  --onnx ./models/t_multi_class.onnx \
  --output ./models/transformer_multi_class_model.rknn \
  --target rk3588 \
  --copy-meta
```

5. 推理时主程序会优先读取 RKNN 同名 sidecar（`*.meta.json`），自动显示攻击类型名称，而不是仅显示类别索引数字。

## ⚙️ 配置说明

### 主程序配置

主程序通过命令行参数配置，无需修改代码中的硬编码路径。

常用参数：
- `-m/--model`：RKNN 模型路径（必填）
- `-c/--config`：预处理配置路径
- `--model-meta`：模型元信息 JSON 路径（可选，默认自动找同名 sidecar）
- `--class-info`：外部类别信息 JSON（可选）
- `--class-labels`：手工标签覆盖（逗号分隔）
- `--benign-label`：良性标签名
- `--benign-class-index`：良性类别索引
- `--binary-threshold`：二分类阈值
- `--auto-align-config`：按模型期望维度自动对齐配置
- `--allow-dim-mismatch`：允许维度不一致时强制运行（不推荐）

### 预处理配置

预处理配置文件（JSON 格式）包含以下信息：
- 数值型特征的归一化参数（最小值、最大值）
- 类别型特征的 Top-K 编码映射
- 滑动窗口大小
- 特征列表

### 模型输入特征

系统支持以下流量特征：

**数值型特征**：
- `IN_PKTS`: 入站数据包数量
- `OUT_PKTS`: 出站数据包数量
- `IN_BYTES`: 入站字节数
- `OUT_BYTES`: 出站字节数
- `FLOW_DURATION_MILLISECONDS`: 流持续时间（毫秒）

**类别型特征**：
- `L4_SRC_PORT`: 源端口号
- `L4_DST_PORT`: 目标端口号
- `PROTOCOL`: 协议类型（TCP/UDP/ICMP等）
- `L7_PROTO`: 应用层协议（HTTP/HTTPS/DNS等）
- `TCP_FLAGS`: TCP 标志位

## ⚠️ 注意事项

1. **权限要求**：实时流量采集需要 root 权限或 CAP_NET_RAW 权限
   ```bash
  sudo python main.py -m ./models/transformer_multi_class_model.rknn --auto-align-config eth0
   ```

2. **网络接口**：确保指定的网络接口存在且处于活动状态

3. **模型兼容性**：确保 RKNN 模型文件与当前硬件平台兼容

4. **metadata 一致性**：建议始终携带模型 sidecar（`*.meta.json`），否则可能只能显示类别索引（如 0/1/2）

5. **滑动窗口**：系统需要累积足够数量的流（达到窗口大小）才开始推理

6. **性能优化**：
   - 调整 `idle_timeout` 参数以平衡检测延迟和准确性
   - 根据硬件性能调整 `window_size` 参数

7. **异常处理**：程序会自动处理预处理和推理过程中的异常，并输出错误信息

## 🔗 Python共享内存 + C++ RKNN推理

如果你希望把 Python 端提取/预处理后的特征交给 C++ 端做 RKNN 推理，可以使用以下流程。

### 1. 编译 C++ 共享内存消费者

```bash
mkdir -p build && cd build
cmake ..
make -j
```

会生成可执行文件：`ai_ids_shm_consumer`

### 2. 启动 C++ 端（读取共享内存并推理）

```bash
./ai_ids_shm_consumer \
  ../models/transformer_multi_class_model.rknn \
  /dev/shm/ai_ids_feature_shm \
  2097152
```

参数说明：
- 第1个参数：RKNN模型路径（可选，默认 `models/transformer_multi_class_model.rknn`）
- 第2个参数：共享内存文件路径（可选，默认 `/dev/shm/ai_ids_feature_shm`）
- 第3个参数：共享内存大小字节数（可选，默认 `2097152`）

### 3. 启动 Python 端（提取特征并写入共享内存）

```bash
python flow_collection/flow_to_shm.py \
  --interface eth0 \
  --config config/config.json \
  --shm-path /dev/shm/ai_ids_feature_shm \
  --shm-size 2097152
```

说明：
- Python 脚本会实时抓流、做滑窗预处理，并按模型输入顺序写入共享内存。
- C++ 端通过序列号检测新数据，读取后直接调用 RKNN 推理。
- 建议先启动 C++ 端，再启动 Python 端。

## 📝 许可证

请查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过 Issue 反馈。
