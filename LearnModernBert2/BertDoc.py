MODERNBERT_INPUTS_DOCSTRING = r"""
Args:
    input_ids (torch.LongTensor, optional): 输入序列的token索引 (batch_size, sequence_length)。使用Flash Attention 2.0时忽略填充。
    attention_mask (torch.Tensor, optional): 注意力掩码，0表示忽略，1表示关注 (batch_size, sequence_length)。
    sliding_window_mask (torch.Tensor, optional): 滑动窗口掩码，用于局部注意力。
    position_ids (torch.LongTensor, optional): 位置索引 (batch_size, sequence_length)。
    inputs_embeds (torch.FloatTensor, optional):  输入嵌入向量 (batch_size, sequence_length, hidden_size)。
    indices (torch.Tensor, optional): 非填充token索引 (total_unpadded_tokens,)。
    cu_seqlens (torch.Tensor, optional): 累积序列长度 (batch + 1,)。
    max_seqlen (int, optional): 最大序列长度 (不含填充)。
    batch_size (int, optional): 批次大小。
    seq_len (int, optional): 序列长度 (含填充)。
    output_attentions (bool, optional): 是否输出注意力权重。
    output_hidden_states (bool, optional): 是否输出隐藏状态。
    return_dict (bool, optional): 是否返回字典格式输出。
"""

MODERNBERT_START_DOCSTRING = r"""
继承自PreTrainedModel。此模型也是PyTorch nn.Module子类。

参数:
    config (ModernBertConfig): 模型配置。
"""
_CHECKPOINT_FOR_DOC = "answerdotai/ModernBERT-base"
_CONFIG_FOR_DOC = "ModernBertConfig"