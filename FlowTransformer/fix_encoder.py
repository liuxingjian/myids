import re

with open("/home/kali/workplace/FlowTransformer/ai-ids-master/FlowTransformer_MultiClassification_Extension/implementations/transformers/basic/encoder_block.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the GPT3Attention init
content = content.replace("GPT3Attention(num_heads, input_dimension, dropout_rate=0.0)", "GPT3Attention(num_heads, inner_dimension, dropout_rate=0.0)")

# Add projection to init
if "self.input_projection = " not in content:
    patch1 = """
        self.input_projection = layers.Dense(inner_dimension, name=f"{prefix}input_projection") if inner_dimension != input_dimension else None
        
        self.attn_implementation = attn_implementation
"""
    content = content.replace("self.attn_implementation = attn_implementation", patch1)

# Add projection to call
if "x = self.input_projection(x)" not in content:
    patch2 = """    def call(self, inputs, training, mask=None):
        x = inputs
        if self.input_projection is not None:
            x = self.input_projection(x)
"""
    content = content.replace("    def call(self, inputs, training, mask=None):\n        x = inputs\n", patch2)
    
# Change inputs + attention_output to x + attention_output
content = content.replace("x = inputs + attention_output", "x = x + attention_output")

# Change the feed forwards to expect inner_dimension instead of input_dimension for feed_forward_1
content = content.replace('name=f"{prefix}feed_forward_1") \\\n            if use_conv else Dense(input_dimension,', 'name=f"{prefix}feed_forward_1") \\\n            if use_conv else Dense(inner_dimension,')
content = content.replace('Conv1D(filters=input_dimension,', 'Conv1D(filters=inner_dimension,')

with open("/home/kali/workplace/FlowTransformer/ai-ids-master/FlowTransformer_MultiClassification_Extension/implementations/transformers/basic/encoder_block.py", "w", encoding="utf-8") as f:
    f.write(content)
