import re

with open("/home/kali/workplace/FlowTransformer/ai-ids-master/FlowTransformer_MultiClassification_Extension/implementations/transformers/basic/encoder_block.py", "r", encoding="utf-8") as f:
    content = f.read()

patch2 = """    def call(self, inputs, training, mask=None):
        if self.input_projection is not None:
            inputs = self.input_projection(inputs)
            
        x = inputs
        x = self.attention(x, x) if self.attn_implementation == MultiHeadAttentionImplementation.Keras else self.attention(x, x, x, mask)

        attention_output = self.attention_dropout(x, training=training) if self.dropout_rate > 0 else x

        x = inputs + attention_output
        x = self.attention_layer_norm(x)
        
        ff_input = x
        x = self.feed_forward_0(x)
        x = self.feed_forward_1(x)
        x = self.feed_forward_dropout(x, training=training) if self.dropout_rate > 0 else x
        feed_forward_output = x

        return self.feed_forward_layer_norm(ff_input + feed_forward_output)
"""

# Replace the whole call function
import re
new_content = re.sub(r'    def call\(self, inputs, training, mask=None\):.*?return self\.feed_forward_layer_norm\(attention_output \+ feed_forward_output\)', patch2, content, flags=re.DOTALL)

with open("/home/kali/workplace/FlowTransformer/ai-ids-master/FlowTransformer_MultiClassification_Extension/implementations/transformers/basic/encoder_block.py", "w", encoding="utf-8") as f:
    f.write(new_content)
