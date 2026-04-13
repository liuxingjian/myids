# FlowTransformer Multi/Binary Classification Extension

## 说明
这是基于 FlowTransformer 的二分类/多分类扩展，重点是让训练产物可直接用于主工程实时推理，并且在推理阶段输出真实类别名（而不是仅输出 0/1/2）。

主要能力：
1. 支持二分类与多分类训练、评估。
2. 评估阶段输出各类别指标。
3. 导出 ONNX 时同步导出模型元信息（输入顺序、类别维度、类别标签映射）。
4. 支持训练机与转换机分离：训练机导出 ONNX，转换机转 RKNN。

原项目请参考：
- https://github.com/liamdm/FlowTransformer

论文信息：
- 标题: FlowTransformer: A Transformer Framework for Flow-based Network Intrusion Detection Systems
- 链接: https://www.sciencedirect.com/science/article/pii/S095741742303066X

## 快速开始

### 1. 训练
- 二分类示例: `binary_classification_demo.ipynb`
- 多分类示例: `multi_classification_demo.ipynb`

### 2. 导出（在训练机）
运行 `multi_classification_demo.ipynb` 的导出单元后，会生成：
- `../models/t_multi_class.onnx`
- `../models/t_multi_class.meta.json`

其中 `t_multi_class.meta.json` 会包含：
- `model_input_names`
- `model_expected_dims`
- `class_map`
- `class_labels`
- `benign_label`
- `benign_class_index`

说明：
- notebook 导出单元会校验 ONNX 权重中是否存在 NaN/Inf。
- 若发现异常会直接中止，避免导出无效模型继续进入 RKNN 转换。

### 3. 转换（在转换机）
在项目根目录执行：

```bash
python tools/onnx_to_rknn.py \
	--onnx ./models/t_multi_class.onnx \
	--output ./models/transformer_multi_class_model.rknn \
	--target rk3588 \
	--copy-meta
```

`--copy-meta` 会把 ONNX 同名元信息复制为 RKNN 同名元信息，供主程序自动读取。

转换后在根目录 `models/` 下会得到：
- `transformer_multi_class_model.rknn`
- `transformer_multi_class_model.meta.json`

### 4. 推理（主工程）
在项目根目录执行：

```bash
python main.py -m ./models/transformer_multi_class_model.rknn --auto-align-config
```

若元信息完整，日志会显示类别名，例如：

```text
Predicted label: Benign, confidence: 0.9766 (class_index=0)
```

而不是仅显示：

```text
Predicted class: 0
```

## 常见问题

### 1. 为什么还是显示 0/1/2？
通常是模型元信息中没有类别映射字段。请检查：
- `models/transformer_multi_class_model.meta.json` 是否包含 `class_labels` 或 `class_map`。

### 2. ONNX 正常导出但 RKNN 推理接近均匀分布怎么办？
优先检查：
1. ONNX initializer 是否存在 NaN/Inf。
2. 推理侧是否读取了正确的 `model_input_names` 与 `model_expected_dims`。
3. 训练/推理使用的预处理配置是否一致。

### 3. 多分类与二分类都支持吗？
支持。主程序会根据输出维度自动解释：
- 单输出按二分类概率处理。
- 多输出按多分类处理，并结合 `benign_label` / `benign_class_index` 判定是否为攻击。

## 目录说明

核心文件：
- `binary_classification_demo.ipynb`: 二分类训练示例。
- `multi_classification_demo.ipynb`: 多分类训练与导出示例。
- `framework/flow_transformer_binary_classification.py`: 二分类流程。
- `framework/flow_transformer_multi_classification.py`: 多分类流程。

## License
This project is based on FlowTransformer and licensed under the AGPL-3.0 License.
