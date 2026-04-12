#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import os

import pandas as pd

from framework.dataset_specification import NamedDatasetSpecifications, DatasetSpecification
from framework.enumerations import EvaluationDatasetSampling
from framework.flow_transformer import FlowTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from framework.framework_component import FunctionalComponent
from implementations.classification_heads import *
from implementations.input_encodings import *
from implementations.pre_processings import StandardPreProcessing
from implementations.transformers.basic_transformers import BasicTransformer
from implementations.transformers.named_transformers import *

encodings = [
    NoInputEncoder(),
    RecordLevelEmbed(64),
    CategoricalFeatureEmbed(EmbedLayerType.Dense, 16),
    CategoricalFeatureEmbed(EmbedLayerType.Lookup, 16),
    CategoricalFeatureEmbed(EmbedLayerType.Projection, 16),
    RecordLevelEmbed(64, project=True)
]

classification_heads = [
    LastTokenClassificationHead(),
    FlattenClassificationHead(),
    GlobalAveragePoolingClassificationHead(),
    CLSTokenClassificationHead(),
    FeaturewiseEmbedding(project=False),
    FeaturewiseEmbedding(project=True),
]

transformers: List[FunctionalComponent] = [
    BasicTransformer(2, 128, n_heads=2),
    BasicTransformer(2, 128, n_heads=2, is_decoder=True),
    GPTSmallTransformer(),
    BERTSmallTransformer()
]

flow_file_path = r"data"

custom_dataset_spec = DatasetSpecification(
    include_fields=['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'FLOW_DURATION_MILLISECONDS', 'OUT_PKTS', 'OUT_BYTES', 'IN_PKTS', 'IN_BYTES', 'L7_PROTO', 'TCP_FLAGS'],
    categorical_fields=['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'TCP_FLAGS'],
    class_column="Attack",
    benign_label="Benign"
)

datasets = [
    ("Balanced_Dataset", os.path.join(flow_file_path, "balanced_dataset.csv"), custom_dataset_spec, 0.2, EvaluationDatasetSampling.RandomRows)
]

pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Define the transformer
ft = FlowTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[0],
                     sequential_model=transformers[0],
                     classification_head=classification_heads[0],
                     params=FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
m = ft.build_model()
m.summary()

# Compile the model
m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'], jit_compile=True)

keras_save_path = os.path.join("..", "models", "my_flow_transformer.h5")

if os.path.exists(keras_save_path):
    print(f"\n--> 发现已保存的 Keras 模型 {keras_save_path}，跳过训练，直接加载...")
    m.load_weights(keras_save_path)
else:
    print("\n--> 未找到已保存的模型，开始模型训练...")
    # Get the evaluation results
    eval_results: pd.DataFrame
    (train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=5, steps_per_epoch=64, early_stopping_patience=5)
    print(eval_results)
    
    m.save(keras_save_path)
    print(f"--> Keras 模型已保存至 {keras_save_path}")

# 1. 临时导出 Keras 模型为 ONNX 模型
print("\n--> 1. 正在将 Keras 模型导出为 ONNX 格式...")
import tf2onnx
onnx_save_path = os.path.join("..", "models", "t.onnx")
# tf2onnx.convert.from_keras 将自动推断 Keras 模型的输入形状
model_proto, _ = tf2onnx.convert.from_keras(m, output_path=onnx_save_path, opset=13)
print(f"ONNX 模型已保存至 {onnx_save_path}")

# 2. 将 ONNX 转化为 RKNN
print("\n--> 2. 正在将 ONNX 模型转化为 RKNN 格式...")
rknn_success = False
try:
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    
    # 假设平台为 rk3588 (请根据您的开发板修改，如 rk3568 等)
    rknn.config(target_platform='rk3588')
    ret = rknn.load_onnx(model=onnx_save_path)
    
    if ret != 0:
        print('加载 ONNX 模型失败!')
    else:
        ret = rknn.build(do_quantization=False) # 变更为 True 如果需要量化
        if ret != 0:
            print('构建 RKNN 模型失败!')
        else:
            rknn_save_path = os.path.join("..", "models", "transformer_model.rknn")
            ret = rknn.export_rknn(rknn_save_path)
            if ret == 0:
                print(f"转换成功! RKNN 模型已保存至 {rknn_save_path}")
                rknn_success = True
            else:
                print('导出 RKNN 模型失败!')
except ImportError:
    print("【警告】未找到 rknn.api 模块！请确保环境中已安装 rknn-toolkit2。")
except Exception as e:
    print(f"RKNN 转换期间发生异常: {e}")
finally:
    # 3. 智能清理中间文件
    print("\n--> 3. 最终模型清理...")
    if rknn_success:
        # 如果 RKNN 成功，按照要求删除其他模型
        if os.path.exists(onnx_save_path):
            os.remove(onnx_save_path)
            print(f"已删除不再需要的 ONNX 模型: {onnx_save_path}")
        if os.path.exists(keras_save_path):
            os.remove(keras_save_path)
            print(f"已删除不再需要的 Keras H5 模型: {keras_save_path}")
        print("清理完毕，当前仅保留最终的 RKNN 模型！")
    else:
        # 如果转换 RKNN 失败，保留 ONNX 以供跨平台/离线手动转换
        print(f"由于 RKNN 转换未成功(可能是缺少开发库)，系统为您保留了 ONNX 模型: {onnx_save_path}")
        print("如果需要在 Linux/其他工具中转换，可以直接使用这个生成的 ONNX 文件！")