from BertTokenizer import SimpleTokenizer
from BertModel import ModernBertModel
import torch

from BertModel import ModernBertConfig
from BertEncoderAttention import AttentionConfig
from BertEmbedding import EmbeddingConfig
from BertEncoderFFN import FFNConfig

embed_config = EmbeddingConfig()
attention_config = AttentionConfig()
ffn_config = FFNConfig()
config = ModernBertConfig(embed_config=embed_config,
                          attention_config=attention_config,
                          ffn_config=ffn_config,
                          local_attention_size=8)


sentence_list = ["The cat sat on the mat, contentedly purring.",
                 "Quantum physics is a fascinating but complex field of study.",
                 "She baked a delicious chocolate cake for her birthday.",
                 "Rain.",
                 "The ancient ruins whispered stories of forgotten empires.",
                 "He meticulously crafted a miniature wooden sailboat.",
                 "Tomorrow's forecast predicts scattered thunderstorms.",
                 "A single red rose bloomed in the desolate garden.",
                 "Understanding human behavior is a lifelong pursuit.",
                 "Despite the challenges, she persevered and ultimately succeeded."]

# 初始化 tokenizer
tokenizer = SimpleTokenizer(max_len=20)
# 对句子进行 tokenize 并 padding
token_index, attention_mask = tokenizer.encode(sentence_list)
# 解码
decoded_sentences = tokenizer.decode(token_index)

# 打印结果
print("Input IDs:")
print(token_index)
print("Attention Mask:")
print(attention_mask)
print("Decoded Sentences:")
print(decoded_sentences)
# 定义 ModernBert 配置

# 创建 ModernBert 模型实例
model = ModernBertModel(config)

# 进行一次 forward 传播
with torch.no_grad():
    outputs = model.forward(input_index=token_index,
                            attention_mask=attention_mask)

# 输出结果
print(outputs.shape)  # 打印形状验证
