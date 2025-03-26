import math
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (add_code_sample_docstrings,
                                add_start_docstrings,
                                add_start_docstrings_to_model_forward,
                                logging)
from transformers.utils.import_utils import is_triton_available
from BertConfig import ModernBertConfig
from BertDoc import MODERNBERT_START_DOCSTRING, MODERNBERT_INPUTS_DOCSTRING, _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC
from BertEmbedding import ModernBertEmbeddings
from BertEncoder import ModernBertEncoderLayer, ModernBertAttention, ModernBertMLP
from BertDecoder import ModernBertPredictionHead
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput, TokenClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from contextlib import nullcontext


logger = logging.get_logger(__name__)


def _unpad_modernbert_input(inputs: torch.Tensor,
                            attention_mask: torch.Tensor,
                            position_ids: Optional[torch.Tensor] = None,
                            labels: Optional[torch.Tensor] = None,
                            ) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    从输入序列中移除填充。
    参数:
        inputs: (batch, seqlen, ...) 或 (batch, seqlen)
        attention_mask: (batch, seqlen), bool / int, 1 表示有效，0 表示无效。
        position_ids: (batch, seqlen), int, 位置 ID。
        labels: (batch, seqlen), int, 标签。
    返回:
        unpadded_inputs: (total_nnz, ...), 其中 total_nnz = attention_mask 中选择的标记数。
        indices: (total_nnz)
        cu_seqlens: (batch + 1), 累积序列长度。
        max_seqlen_in_batch: int
        unpadded_position_ids: (total_nnz) 或 None
        unpadded_labels: (total_nnz) 或 None
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = position_ids.flatten()[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels


def _pad_modernbert_output(inputs: torch.Tensor,
                           indices: torch.Tensor,
                           batch: int,
                           seqlen: int,
                           ) -> torch.Tensor:
    """
    为序列添加填充。
    参数:
        inputs: (total_nnz, ...) 或 (total_nnz,), total_nnz 为 attention_mask 中选择的标记数。
        indices: (total_nnz)
        batch: int, 批次大小。
        seqlen: int, 最大序列长度。
    返回:
        padded_inputs: (batch, seqlen, ...) 或 (batch, seqlen)
    """
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    return padded_inputs




@add_start_docstrings("裸露的 ModernBert 模型输出原始隐藏状态，没有任何特定的头部。",
                      MODERNBERT_START_DOCSTRING,
                      )
class ModernBertPreTrainedModel(PreTrainedModel):
    config_class = ModernBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModernBertEmbeddings", "ModernBertEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False

    def _init_weights(self, module: nn.Module):
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
            nn.init.trunc_normal_(module.weight,
                                  mean=0.0,
                                  std=std,
                                  a=-cutoff_factor * std,
                                  b=cutoff_factor * std,
                                  )
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size ** -0.5,
        }
        if isinstance(module, ModernBertEmbeddings):
            init_weight(module.tok_embeddings, stds["embedding"])
        elif isinstance(module, ModernBertMLP):
            init_weight(module.Wi, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertAttention):
            init_weight(module.Wqkv, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertPredictionHead):
            init_weight(module.dense, stds["out"])
        elif isinstance(module, ModernBertForMaskedLM):
            init_weight(module.decoder, stds["out"])
        elif isinstance(module, (ModernBertForSequenceClassification, ModernBertForTokenClassification)):
            init_weight(module.classifier, stds["final_out"])

    @classmethod
    def _autoset_attn_implementation(cls,
                                     config,
                                     use_flash_attention_2: bool = False,
                                     torch_dtype: Optional[torch.dtype] = None,
                                     device_map: Optional[Union[str, Dict[str, int]]] = None,
                                     check_device_map: bool = True,
                                     ):
        # 如果用户没有指定任何内容，尝试使用 flash_attention_2（如果可用）。
        # 否则我们回退到超类方法中的默认 SDPA -> Eager。
        # ModernBert 的 FA2 实现正确处理非 fp16/bf16 数据类型，我们不需要非 fp16/bf16 数据类型的 FA2 警告，因此我们为 FA2 检查设置 fp16。
        if config._attn_implementation_internal is None:
            config._attn_implementation_internal = "flash_attention_2"
            try:
                return cls._check_and_enable_flash_attn_2(config,
                                                          torch_dtype=torch.float16,
                                                          device_map=device_map,
                                                          hard_check_only=False,
                                                          check_device_map=check_device_map,
                                                          )
            except (ValueError, ImportError):
                config._attn_implementation_internal = None
        return super()._autoset_attn_implementation(config,
                                                    use_flash_attention_2=use_flash_attention_2,
                                                    torch_dtype=torch.float16,
                                                    device_map=device_map,
                                                    check_device_map=check_device_map,
                                                    )

    def _maybe_set_compile(self):
        if self.config.reference_compile is False:
            return

        if hasattr(self, "hf_device_map") and len(self.hf_device_map) > 1:
            if self.config.reference_compile:
                logger.warning_once("如果 `accelerate` 将模型分割到多个设备上，`torch.compile` 将无法工作。回退到非编译模式。"
                                    )
            self.config.reference_compile = False
        if self.device.type == "mps":
            if self.config.reference_compile:
                logger.warning_once("使用 `torch.compile` 编译模型并使用 `torch.mps` 设备是不支持的。回退到非编译模式。")
            self.config.reference_compile = False
        if self.device.type == "cpu":
            if self.config.reference_compile:
                logger.warning_once("使用 `torch.compile` 编译模型并使用 `torch.cpu` 设备是不支持的。回退到非编译模式。")
            self.config.reference_compile = False
        if self.config.reference_compile is None:
            self.config.reference_compile = is_triton_available()

    def resize_token_embeddings(self, *args, **kwargs):
        model_embeds = super().resize_token_embeddings(*args, **kwargs)

        if self.config.reference_compile in {True, None}:
            if self.config.reference_compile:
                logger.warning_once("使用 `torch.compile` 调整令牌嵌入大小是不支持的。回退到非编译模式。")
            self.config.reference_compile = False

        return model_embeds


@add_start_docstrings("The bare ModernBert Model outputting raw hidden-states without any specific head on top.",
                      MODERNBERT_START_DOCSTRING,
                      )
class ModernBertModel(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.encoder_layers = nn.ModuleList()
        for layer_id in range(config.num_hidden_layers):
            encoder_layer = ModernBertEncoderLayer(config, layer_id)
            self.encoder_layers.append(encoder_layer)
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC,
                                output_type=BaseModelOutput,
                                config_class=_CONFIG_FOR_DOC,
                                )
    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                sliding_window_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                indices: Optional[torch.Tensor] = None,
                cu_seqlens: Optional[torch.Tensor] = None,
                max_seqlen: Optional[int] = None,
                batch_size: Optional[int] = None,
                seq_len: Optional[int] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutput]:
        if output_attentions is None:
            output_attentions = self.config.output_attentions

        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        if return_dict is None:
            return_dict = self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if output_hidden_states:
            all_hidden_states = ()
        else:
            all_hidden_states = None

        if output_attentions:
            all_self_attentions = ()
        else:
            all_self_attentions = None

        self._maybe_set_compile()

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]

        if input_ids is not None:
            device = input_ids.device
        else:
            device = inputs_embeds.device

        if attention_mask is None:
            shape = (batch_size, seq_len)
            attention_mask = torch.ones(size=shape, device=device, dtype=torch.bool)

        repad = False
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                repad = True
                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(inputs=input_ids,
                                                                                                 attention_mask=attention_mask
                                                                                                 )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(inputs=inputs_embeds,
                                                                                                 attention_mask=attention_mask
                                                                                                 )
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # attention_mask
            # (batch_size, sequence_length)
            # to (batch_size, 1, sequence_length, sequence_length)
            attention_mask, sliding_window_mask = self.update_attention_mask(attention_mask=attention_mask,
                                                                             output_attentions=output_attentions)

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        for encoder_layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(encoder_layer.__call__,
                                                                  hidden_states,
                                                                  attention_mask,
                                                                  sliding_window_mask,
                                                                  position_ids,
                                                                  cu_seqlens,
                                                                  max_seqlen,
                                                                  output_attentions,
                                                                  )
            else:
                layer_outputs = encoder_layer(hidden_states,
                                              attention_mask=attention_mask,
                                              sliding_window_mask=sliding_window_mask,
                                              position_ids=position_ids,
                                              cu_seqlens=cu_seqlens,
                                              max_seqlen=max_seqlen,
                                              output_attentions=output_attentions,
                                              )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = self.final_norm(hidden_states)

        if repad:
            hidden_states = _pad_modernbert_output(inputs=hidden_states, indices=indices, batch=batch_size,
                                                   seqlen=seq_len
                                                   )
            if all_hidden_states is not None:
                all_hidden_states = tuple(
                    _pad_modernbert_output(inputs=hs, indices=indices, batch=batch_size, seqlen=seq_len)
                    for hs in all_hidden_states
                    )
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states,
                               hidden_states=all_hidden_states,
                               attentions=all_self_attentions,
                               )

    def update_attention_mask(self, attention_mask: torch.Tensor, output_attentions: bool) -> torch.Tensor:
        if output_attentions:
            if self.config._attn_implementation == "sdpa":
                logger.warning_once("Outputting attentions is only supported with the 'eager' attention implementation"
                                    'not with "sdpa". Falling back to `attn_implementation="eager"`.')
                self.config._attn_implementation = "eager"
            elif self.config._attn_implementation != "eager":
                logger.warning_once("Outputting attentions is only supported with the eager attention implementation"
                                    f'not with {self.config._attn_implementation}. '
                                    f'Consider setting `attn_implementation="eager"`.'
                                    " Setting `output_attentions=False`.")

        # attention_mask
        # (batch_size, sequence_length)
        # to (batch_size, 1, sequence_length, sequence_length)
        global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)

        # Create position indices
        rows = global_attention_mask.shape[2]
        rows = torch.arange(rows).unsqueeze(0)
        rows = rows.unsqueeze(0)

        # Calculate distance between positions
        distance = torch.abs(rows - rows.T)

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = distance <= self.config.local_attention // 2
        window_mask = window_mask.unsqueeze(0)
        window_mask = window_mask.unsqueeze(0)
        window_mask = window_mask.to(attention_mask.device)

        # Combine with existing mask
        mask = window_mask.logical_not()
        value = torch.finfo(self.dtype).min
        sliding_window_mask = global_attention_mask.masked_fill(mask=mask, value=value)

        return global_attention_mask, sliding_window_mask

    @add_start_docstrings(
        "The ModernBert Model with a decoder head on top that is used for masked language modeling.",
        MODERNBERT_START_DOCSTRING,
    )
    class ModernBertForMaskedLM(ModernBertPreTrainedModel):
        _tied_weights_keys = ["decoder.weight"]

        def __init__(self, config: ModernBertConfig):
            super().__init__(config)
            self.config = config
            self.model = ModernBertModel(config)
            self.head = ModernBertPredictionHead(config)
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

            self.sparse_prediction = self.config.sparse_prediction
            self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

            # Initialize weights and apply final processing
            self.post_init()

        def get_output_embeddings(self):
            return self.decoder

        def set_output_embeddings(self, new_embeddings: nn.Linear):
            self.decoder = new_embeddings

        @torch.compile(dynamic=True)
        def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.head(output))

        @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=MaskedLMOutput,
            config_class=_CONFIG_FOR_DOC,
        )
        def forward(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                sliding_window_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                indices: Optional[torch.Tensor] = None,
                cu_seqlens: Optional[torch.Tensor] = None,
                max_seqlen: Optional[int] = None,
                batch_size: Optional[int] = None,
                seq_len: Optional[int] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
        ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            self._maybe_set_compile()

            if self.config._attn_implementation == "flash_attention_2":
                if indices is None and cu_seqlens is None and max_seqlen is None:
                    if batch_size is None and seq_len is None:
                        if inputs_embeds is not None:
                            batch_size, seq_len = inputs_embeds.shape[:2]
                        else:
                            batch_size, seq_len = input_ids.shape[:2]
                    device = input_ids.device if input_ids is not None else inputs_embeds.device

                    if attention_mask is None:
                        attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

                    if inputs_embeds is None:
                        with torch.no_grad():
                            input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                                inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                labels=labels
                            )
                    else:
                        inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                            inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
                            labels=labels
                        )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                batch_size=batch_size,
                seq_len=seq_len,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            last_hidden_state = outputs[0]

            if self.sparse_prediction and labels is not None:
                # flatten labels and output first
                labels = labels.view(-1)
                last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

                # then filter out the non-masked tokens
                mask_tokens = labels != self.sparse_pred_ignore_index
                last_hidden_state = last_hidden_state[mask_tokens]
                labels = labels[mask_tokens]

            logits = (
                self.compiled_head(last_hidden_state)
                if self.config.reference_compile
                else self.decoder(self.head(last_hidden_state))
            )

            loss = None
            if labels is not None:
                loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size)

            if self.config._attn_implementation == "flash_attention_2":
                with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                    logits = _pad_modernbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)

            if not return_dict:
                output = (logits,)
                return ((loss,) + output) if loss is not None else output

            return MaskedLMOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

@add_start_docstrings(
    "The ModernBert Model with a sequence classification head on top that performs pooling.",
    MODERNBERT_START_DOCSTRING,
)
class ModernBertForSequenceClassification(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            sliding_window_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            indices: Optional[torch.Tensor] = None,
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None,
            batch_size: Optional[int] = None,
            seq_len: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.config.classifier_pooling == "cls":
            last_hidden_state = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "The ModernBert Model with a token classification head on top, e.g. for Named Entity Recognition (NER) tasks.",
    MODERNBERT_START_DOCSTRING,
)
class ModernBertForTokenClassification(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            sliding_window_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            indices: Optional[torch.Tensor] = None,
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None,
            batch_size: Optional[int] = None,
            seq_len: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    "The ModernBert Model with a decoder head on top that is used for masked language modeling.",
    MODERNBERT_START_DOCSTRING,
)
class ModernBertForMaskedLM(ModernBertPreTrainedModel):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(output))

    @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device if input_ids is not None else inputs_embeds.device

                if attention_mask is None:
                    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                    )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.decoder(self.head(last_hidden_state))
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size)

        if self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_modernbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



