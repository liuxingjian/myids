import tensorflow as tf
from framework.flow_transformer_multi_classification import FlowTransformer
from implementations.transformers.basic_transformers import BasicTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from implementations.classification_heads import LastTokenClassificationHead
from implementations.input_encodings import NoInputEncoder
from implementations.pre_processings import StandardPreProcessing

pp = StandardPreProcessing(n_categorical_levels=32)
ft = FlowTransformer(pre_processing=pp,
                     input_encoding=NoInputEncoder(),
                     sequential_model=BasicTransformer(2, 128, n_heads=2),
                     classification_head=LastTokenClassificationHead(),
                     params=FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

m = ft.build_model()
print("Model built")
