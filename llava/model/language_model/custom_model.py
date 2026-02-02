import math
import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.generation.utils import GenerateOutput
from transformers import AutoConfig, AutoModelForCausalLM

from .llava_llama import LlavaLlamaModel
from ..llava_arch import LlavaMetaForCausalLM
from .llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from llava.train_utils import compute_sum_of_weights, initialize_lstm_weights


def print_hook(grad):
    print(f"Gradient computed for tensor: Sum={grad.sum().item()}", grad.shape)
        


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h, c):
        combined = torch.cat((x, h), 1)

        i = torch.sigmoid(self.W_i(combined))
        f = torch.sigmoid(self.W_f(combined))
        o = torch.sigmoid(self.W_o(combined))
        c_hat = torch.tanh(self.W_c(combined))

        c = f * c + i * c_hat
        h = o * torch.tanh(c)

        return h, c
    
class FactorizedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, rank: int = None):
        super(FactorizedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        
        print(f"[FactorizedLSTMCell] Rank: {rank}, input_size: {input_size}, hidden_size: {hidden_size}")
        
        if rank is None:
            self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
            
        # Factorized weight matrices
        else:
            print(f"[FactorizedLSTMCell] Using factorized weight matrices")
            self.W_i = nn.Sequential(
                nn.Linear(input_size + hidden_size, rank, bias=False),
                nn.Linear(rank, hidden_size, bias=True)
            )
            
            self.W_f = nn.Sequential(
                nn.Linear(input_size + hidden_size, rank, bias=False),
                nn.Linear(rank, hidden_size, bias=True)
            )
            
            self.W_c = nn.Sequential(
                nn.Linear(input_size + hidden_size, rank, bias=False),
                nn.Linear(rank, hidden_size, bias=True)
            )
            
            self.W_o = nn.Sequential(
                nn.Linear(input_size + hidden_size, rank, bias=False),
                nn.Linear(rank, hidden_size, bias=True)
            )
        
        
        

    def init_weights(self):
        with torch.no_grad():
            
            for name, param in self.named_parameters():
                print(name, param.shape)
                if param.numel() > 1:
                    if 'weight' in name and "bias" not in name:
                        # Using numpy
                        # limit = np.sqrt(6 / (param.size(0) + param.size(1)))
                        # param.uniform_(-limit, limit)
                        
                        # Using PyTorch
                        nn.init.xavier_uniform_(param.weight)
                        
                    elif 'bias' in name:
                        param.fill_(0)
                else:
                    print(f"Skipping {name} with shape {param.shape}")
        
    def forward(self, x, h, c):
        combined = torch.cat((x, h), 1)
        
        i = torch.sigmoid(self.W_i(combined))
        f = torch.sigmoid(self.W_f(combined))
        o = torch.sigmoid(self.W_o(combined))
        c_hat = torch.tanh(self.W_c(combined))
        
        c = f * c + i * c_hat
        h = o * torch.tanh(c)
        
        return h, c

class CustomLlavaConfig(LlavaConfig):
    model_type = "custom_llava_llama"


class CustomLlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = CustomLlavaConfig

    def __init__(self, config, query_size: int=1, num_heads: int = 8, lstm_num_layers: int = 2, lstm_bidirectional: bool = False):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.hidden_size = config.hidden_size
        
        self.query_size = query_size
        
        print(f"[CustomLlavaLlamaForCausalLM] Query size: {self.query_size}, hidden size: {self.hidden_size}")
        
        
        
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = lstm_bidirectional
        
        self.lstm_cell = FactorizedLSTMCell(self.query_size * config.hidden_size, self.query_size * config.hidden_size)

        # self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=lstm_num_layers, batch_first=True, bidirectional=lstm_bidirectional)
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, num_layers=2, batch_first=True, bidirectional=False)
        
        
        
        if self.query_size > 1:
            # Define fixed query as a learnable parameter
            self.memory_fixed_Q = nn.Parameter(torch.randn(query_size, 1, self.hidden_size))
            # Trainable attention layers for projection
            self.memory_attention_fixed = nn.MultiheadAttention(config.hidden_size, num_heads)
            # Attention mechanism to query useful information from self.hidden[0]
            self.hidden_attention = nn.MultiheadAttention(config.hidden_size, num_heads)
        
        else:
            self.W_q = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        
        # for name, param in self.named_parameters():
        #     if "memory" in name or "lstm" in name:
        #         param.register_hook(print_hook)
        
        self.alpha = 0.995
        self.step_count = 0
        
        # Initialize weights and apply final processing
        self.post_init()
        


    def get_model(self):
        return self.model
    
    
    def init_weights(self):
        with torch.no_grad():
            for name, param in self.lstm_cell.named_parameters():
                
                if param.numel() > 1:
                
                    if 'weight_' in name or 'W_' in name and not "bias" in name:
                        # limit = torch.sqrt(torch.tensor(6.0 / (param.size(0) + param.size(1))))
                        print(f"Initializing {name} with shape {param.shape}")
                        nn.init.xavier_uniform_(param)
                        # param.uniform_(-limit, limit)
                        
                        # limit = np.sqrt(6 / (param.size(0) / 4 + param.size(1)))
                        # np_weights = np.random.uniform(-limit, limit, param.shape)
                        # param.data = torch.from_numpy(np_weights).to(param.data.dtype)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        
                else:
                    print(f"Skipping {name} with shape {param.shape}")
                    
            if hasattr(self, "memory_fixed_Q"):
                for param in [self.memory_fixed_Q]:
                    if param.numel() > 1:
                        nn.init.xavier_uniform_(self.memory_fixed_Q)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        reset_state: Optional[bool] = True,
        last_clips: Optional[torch.BoolTensor] = None,
        ids: Optional[List] = None,
        cache_position=None, # Added for compatibility
        interpolated_embeds=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        
        # During training, sequentially feed the input_ids belonging to the same video
        if isinstance(input_ids, list):
            # print("Input IDs")
            # print(ids)
            
            memory = torch.zeros((input_ids[i].shape[0], len(input_ids), self.hidden_size), device=input_ids[0].device, dtype=input_ids[0].dtype)
        
            for i in range(len(input_ids)):
                input_ids_step = input_ids[i]
                
                # Project hidden states to (B, fixed_length, 4096)  
                B, L_t = input_ids_step.shape # Shape: (B, length_at_t)
                
                position_ids_step = position_ids[i] if position_ids is not None else None
                attention_mask_step = attention_mask[i]
                past_key_values_step = past_key_values[i] if past_key_values is not None else None
                labels_step = labels[i] if labels is not None else None
                images_step = images[i] if images is not None else None
                image_sizes_step = image_sizes[i] if image_sizes is not None else None
                inputs_embeds_step = inputs_embeds[i] if inputs_embeds is not None else None
                
                if inputs_embeds_step is None:
                    (
                        input_ids_step,
                        position_ids_step,
                        attention_mask_step,
                        past_key_values_step,
                        inputs_embeds_step,
                        labels_step
                    ) = self.prepare_inputs_labels_for_multimodal(
                        input_ids_step,
                        position_ids_step,
                        attention_mask_step,
                        past_key_values,
                        labels_step,
                        images_step,
                        image_sizes_step
                    )
                
                    
                if i == 0:

                    lstm_h = torch.zeros(B, self.hidden_size * self.query_size, device=input_ids[0].device, dtype=inputs_embeds_step.dtype)
                    lstm_c = torch.zeros(B, self.hidden_size * self.query_size, device=input_ids[0].device, dtype=inputs_embeds_step.dtype)
                    
                    
                else:
                    # Extract the last hidden state (not cell state)
                    #hidden_state = self.hidden[0] # TODO: Change this  # Shape: (2, batch, hidden_size)
                    # hidden_state = hidden_state.permute(1, 0, 2).reshape(hidden_state.size(1), -1, hidden_state.size(2))  # Shape: (batch, 2, hidden_size)
                    
                    # Query useful information from hidden state
                    # inputs_embeds_step: (seq_len, batch, hidden_size)
                    attn_output, _ = self.hidden_attention(inputs_embeds_step.permute(1, 0, 2), lstm_h.reshape(self.query_size, B, self.hidden_size), lstm_h.reshape(self.query_size, B, self.hidden_size))
                    attn_output = attn_output.permute(1, 0, 2)  # Shape: (batch, seq_len, hidden_size)
                    
                    inputs_embeds_step = self.alpha * inputs_embeds_step + (1 - self.alpha) * attn_output


                outputs = super(CustomLlavaLlamaForCausalLM, self).forward(
                    input_ids=input_ids_step,
                    attention_mask=attention_mask_step,
                    position_ids=position_ids_step,
                    past_key_values=past_key_values_step,
                    inputs_embeds=inputs_embeds_step,
                    labels=labels_step,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True, # This should always be true
                    return_dict=return_dict
                )
                
                
                if self.args.aggregated_embeds:
                
                    h = outputs.hidden_states[-1]  # Shape: (B, length_at_t, 4096)
                
                if self.query_size == 1:
                    h = h.mean(dim=1)  # Shape: (B, 4096)
                    memory[:, i, :] = h
                    
                    # Use the current aggregated hidden state as the query to query useful information from previou hidden states
                    Q = self.W_q(h)  # Shape: (B, 4096)
                    
                    fixed_output, _ = self.memory_attention_fixed(Q, fixed_key, fixed_value) # (fixed_length, B, 4096)
                    
                else:
                
                
                    # Use fixed query to query useful information from the hidden states
                    memory_fixed_Q = self.memory_fixed_Q.expand(-1, B, -1)  # Shape: (fixed_length, B, 4096)
                    fixed_key = h.permute(1, 0, 2)  # Shape: (length_at_t, B, 4096)
                    fixed_value = h.permute(1, 0, 2)  # Shape: (length_at_t, B, 4096)

                    fixed_output, _ = self.memory_attention_fixed(memory_fixed_Q, fixed_key, fixed_value) # (fixed_length, B, 4096)
                    # fixed_output = fixed_output.permute(1, 0, 2)  # Shape: (B, fixed_length, 4096)
                    fixed_output = fixed_output.reshape(B, -1)  # Shape: (B, fixed_length * 4096)
                    

                if fixed_output.isnan().any():
                    raise ValueError("NaN detected in attention outputs (`fixed_output`)")
                
                # Information from the previous step is fed the hidden states through LSTM
                
                lstm_h, lstm_c = self.lstm_cell(fixed_output.reshape(B, self.query_size * self.hidden_size), lstm_h, lstm_c)
                
                if lstm_h.isnan().any():
                    raise ValueError("NaN detected in hidden states")
            
          
        else:
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )

            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict
            )
        # output.hidden_states: list of tensors with shape (B, length, 4096)
        self.step_count += 1
        
        if self.training:
            for name, module in {"memory_fixed_Q": self.memory_fixed_Q, 
                                "memory_attention_fixed": self.memory_attention_fixed,
                                "hidden_attention": self.hidden_attention,
                                "lstm": self.lstm_cell}.items():
                sum_of_weights = compute_sum_of_weights(module)
                print(f"{name}: {sum_of_weights}")
            
        return outputs
        

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        step: int = 0,
        lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        
        for n, p in self.named_parameters():
            if p.norm().isinf():
                raise ValueError(f"{n} has infinite norm")
            
            if p.isnan().any():
                raise ValueError(f"{n} has NaN values")
        
        if isinstance(inputs, torch.Tensor):

            if images is not None:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)
                
                
            
            B, L_t = inputs_embeds.shape[:2]
            
            if step == 0:
                assert lstm_states is None
                lstm_h = torch.zeros(B, self.hidden_size * self.query_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                lstm_c = torch.zeros(B, self.hidden_size * self.query_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                
            else:
                lstm_h, lstm_c = lstm_states
                attn_output, _ = self.hidden_attention(inputs_embeds.permute(1, 0, 2), lstm_h.reshape(self.query_size, B, self.hidden_size), lstm_h.reshape(self.query_size, B, self.hidden_size))
                attn_output = attn_output.permute(1, 0, 2)  # Shape: (batch, seq_len, hidden_size)
                
                print(f"[Norm] (lstm_h): {lstm_h.norm().item()}")
                print(f"[Norm] (attn_output): {attn_output.norm().item()}")
                
                inputs_embeds = self.alpha * inputs_embeds + (1 - self.alpha) * attn_output
            
            outputs = super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                return_dict_in_generate=True, 
                output_hidden_states=True,
                **kwargs
            )
            
            h = outputs.hidden_states[0][-1]
            
            if h.isnan().any():
                raise ValueError(f"[Step={step}] NaN detected in hidden states")
            
        
            
            # Shape: (B, length_at_t - 1, 4096)
            # Use fixed query to query useful information from the hidden states
            memory_fixed_Q = self.memory_fixed_Q.expand(-1, B, -1)  # Shape: (fixed_length, B, 4096)
            fixed_key = h.permute(1, 0, 2)  # Shape: (length_at_t, B, 4096)
            fixed_value = h.permute(1, 0, 2)  # Shape: (length_at_t, B, 4096)

            fixed_output, _ = self.memory_attention_fixed(memory_fixed_Q, fixed_key, fixed_value) # (fixed_length, B, 4096)
            # fixed_output = fixed_output.permute(1, 0, 2)  # Shape: (B, fixed_length, 4096)
            fixed_output = fixed_output.reshape(B, -1)  # Shape: (B, fixed_length * 4096)
            lstm_h, lstm_c = self.lstm_cell(fixed_output.reshape(B, self.query_size * self.hidden_size), lstm_h, lstm_c)
            
            if lstm_h.isnan().any():
                raise ValueError(f"[Step={step}] NaN detected in hidden states")
            
            return outputs.sequences, (lstm_h, lstm_c)
                 

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    
    
AutoConfig.register("custom_llava_llama", CustomLlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
