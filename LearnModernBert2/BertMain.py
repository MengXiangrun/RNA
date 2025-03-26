from BertConfig import ModernBertConfig
from BertTokenizer import SimpleTokenizer
from BertModel import ModernBertModel
import torch

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
config = ModernBertConfig(vocab_size=len(tokenizer.vocab),
                          hidden_size=768,
                          intermediate_size=3072,
                          num_hidden_layers=2,  # Reduced for faster execution during testing
                          num_attention_heads=12,
                          max_position_embeddings=512,
                          hidden_activation="gelu",
                          norm_eps=1e-5,
                          pad_token_id=tokenizer.word_to_id['[PAD]'],
                          eos_token_id=tokenizer.word_to_id['[SEP]'],
                          bos_token_id=tokenizer.word_to_id['[CLS]'],
                          cls_token_id=tokenizer.word_to_id['[CLS]'],
                          sep_token_id=tokenizer.word_to_id['[SEP]'],
                          global_rope_theta=10000.0,
                          attention_dropout=0.1,
                          global_attn_every_n_layers=3,
                          local_attention=128,
                          local_rope_theta=10000.0,
                          embedding_dropout=0.1,
                          mlp_dropout=0.1,
                          classifier_pooling="cls",
                          classifier_dropout=0.1,
                          classifier_activation="gelu",
                          deterministic_flash_attn=False,
                          sparse_prediction=False,
                          sparse_pred_ignore_index=-100,
                          reference_compile=None,
                          repad_logits_with_grad=False
                          )
# 创建 ModernBert 模型实例
model = ModernBertModel(config)
help(model)

# 进行一次 forward 传播
with torch.no_grad():
    outputs = model.forward(input_ids=token_index,
                            attention_mask=attention_mask)

# 输出结果
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)  # 打印形状验证
