import torch
from tqdm import tqdm, trange
import random
import os
from llava_utils import prompt_wrapper, generator
from torchvision.utils import save_image
import numpy as np
from copy import deepcopy
import time
import json

from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
import seaborn as sns


class Attacker:

    def __init__(self, args, model, tokenizer, targets, device='cuda:0', safe_image=None):

        self.args = args
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"

        self.targets = targets # targets that we want to promte likelihood
        self.loss_buffer = []
        self.num_targets = len(self.targets)
        
        # Store safe image (image + image patch) for joint optimization
        self.safe_image = safe_image

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)

    def get_vocabulary(self):

        vocab_dicts = self.tokenizer.get_vocab()
        vocabs = vocab_dicts.keys()

        single_token_vocabs = []
        single_token_vocabs_embedding = []
        single_token_id_to_vocab = dict()
        single_token_vocab_to_id = dict()

        cnt = 0

        for item in vocabs:
            tokens = self.tokenizer(item, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if tokens.shape[1] == 1:

                single_token_vocabs.append(item)
                emb = self.model.model.embed_tokens(tokens)
                single_token_vocabs_embedding.append(emb)

                single_token_id_to_vocab[cnt] = item
                single_token_vocab_to_id[item] = cnt

                cnt+=1

        single_token_vocabs_embedding = torch.cat(single_token_vocabs_embedding, dim=1).squeeze()

        self.vocabs = single_token_vocabs
        self.embedding_matrix = single_token_vocabs_embedding.to(self.device)
        self.id_to_vocab = single_token_id_to_vocab
        self.vocab_to_id = single_token_vocab_to_id

    def hotflip_attack(self, grad, token,
                       increase_loss=False, num_candidates=1):

        token_id = self.vocab_to_id[token]
        token_emb = self.embedding_matrix[token_id] # embedding of current token

        scores = ((self.embedding_matrix - token_emb) @ grad.T).squeeze(1)

        if not increase_loss:
            scores *= -1  # lower versus increase the class probability.

        _, best_k_ids = torch.topk(scores, num_candidates)
        return best_k_ids.detach().cpu().numpy()

    def wrap_prompt_simple(self, text_prompt_template, adv_prompt, batch_size):

        text_prompts = text_prompt_template + ' ' + adv_prompt # insert the adversarial prompt

        prompt = prompt_wrapper.Prompt(model=self.model, tokenizer = self.tokenizer, text_prompts=[text_prompts])

        prompt.context_embs[0] = prompt.context_embs[0].detach().requires_grad_(True)
        prompt.context_embs = prompt.context_embs * batch_size

        return prompt

    def update_adv_prompt(self, adv_prompt_tokens, idx, new_token):
        next_adv_prompt_tokens = deepcopy(adv_prompt_tokens)
        next_adv_prompt_tokens[idx] = new_token
        next_adv_prompt = ' '.join(next_adv_prompt_tokens)
        return next_adv_prompt_tokens, next_adv_prompt



    def attack(self, text_prompt_template, offset, batch_size = 8, num_iter=2000):

        print('>>> batch_size: ', batch_size)
        
        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)

        self.get_vocabulary()
        vocabs, embedding_matrix = self.vocabs, self.embedding_matrix

        trigger_token_length = getattr(self.args, 'patch_length', 16) # equivalent to
        adv_prompt_tokens = random.sample(vocabs, trigger_token_length)
        adv_prompt = ' '.join(adv_prompt_tokens)
        print(len(vocabs),adv_prompt)

        st = time.time()

        for t in tqdm(range(num_iter+1)):

            for token_to_flip in range(0, trigger_token_length): # for each token in the trigger

                batch_targets = random.sample(self.targets, batch_size)
                prompt = self.wrap_prompt_simple(text_prompt_template, adv_prompt, batch_size)
                if t==0 and token_to_flip==0:
                    print(prompt.text_prompts)

                target_loss = -self.attack_loss(prompt, batch_targets, images=self.safe_image)
                loss = target_loss # to minimize
                loss.backward()

                print('[adv_prompt]', adv_prompt)
                print("target_loss: %f" % (target_loss.item()))
                self.loss_buffer.append(target_loss.item())

                tokens_grad = prompt.context_embs[0].grad[:, token_to_flip+offset, :]
                candidates = self.hotflip_attack(tokens_grad, adv_prompt_tokens[token_to_flip],
                                            increase_loss=False, num_candidates=self.args.n_candidates)

                self.model.zero_grad()

                # try all the candidates and pick the best
                # comparing candidates does not require gradient computation
                with torch.no_grad():
                    curr_best_loss = 999999
                    curr_best_trigger_tokens = None
                    curr_best_trigger = None

                    for cand in candidates:
                        next_adv_prompt_tokens, next_adv_prompt = self.update_adv_prompt(adv_prompt_tokens,
                                                                    token_to_flip, self.id_to_vocab[cand])
                        prompt = self.wrap_prompt_simple(text_prompt_template, next_adv_prompt, batch_size)

                        next_target_loss = -self.attack_loss(prompt, batch_targets, images=self.safe_image)
                        curr_loss = next_target_loss  # to minimize

                        if curr_loss < curr_best_loss:
                            curr_best_loss = curr_loss
                            curr_best_trigger_tokens = next_adv_prompt_tokens
                            curr_best_trigger = next_adv_prompt

                    # Update overall best if the best current candidate is better
                    if curr_best_loss < loss:
                        adv_prompt_tokens = curr_best_trigger_tokens
                        adv_prompt = curr_best_trigger
                print('(update: %f minutes)' % ((time.time() - st) / 60))

            self.plot_loss()

            print('######### Output - Iter = %d ##########' % t)
                
        return adv_prompt
    

    def evaluate_safety_patch(self, text_prompt_template, offset, batch_size = 8, num_iter=2000):
        """
        Evaluate the safety patch by computing the loss of generating harmful contents. 
        
        Created: 2024.9.18

        """

        num_iter = int(np.ceil(len(self.targets) / batch_size))

        print('>>> batch_size: ', batch_size)
        
        my_generator = generator.Generator(model=self.model, tokenizer=self.tokenizer)

        self.get_vocabulary()
        vocabs, embedding_matrix = self.vocabs, self.embedding_matrix

        trigger_token_length = getattr(self.args, 'patch_length', 16) # equivalent to
        adv_prompt_tokens = random.sample(vocabs, trigger_token_length)
        adv_prompt = ' '.join(adv_prompt_tokens)
        print(len(vocabs),adv_prompt)

        st = time.time()

        safety_patches = json.load(open("text_safety_patch_llm_generated.json", 'r'))

        loss_dict = defaultdict(int)

        for t in trange(num_iter, desc="Computing loss"):

            batch_targets = self.targets[batch_size * t: batch_size * (t + 1)]

            for index_safety_patch, safety_patch in enumerate(safety_patches):

                prompt = self.wrap_prompt_simple(text_prompt_template, safety_patch, min(batch_size, len(self.targets) - batch_size * t))


                loss = -self.attack_loss(prompt, batch_targets, images=self.safe_image)

                loss_dict[index_safety_patch] += loss.item()


        print(loss)

        loss_list = list(loss_dict.values())
        
        min_index = loss_list.index(min(loss_list))

        return safety_patches[min_index]
                





    def plot_loss(self):

        sns.set_theme()

        num_iters = len(self.loss_buffer)

        num_iters = min(num_iters, 5000)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer[:num_iters], label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        # Use run_id if available to avoid overwriting, otherwise use default name
        if hasattr(self.args, 'run_id') and self.args.run_id:
            loss_curve_file = os.path.join(self.args.save_dir, f'loss_curve_{self.args.run_id}.png')
            loss_file = os.path.join(self.args.save_dir, f'loss_{self.args.run_id}')
        else:
            loss_curve_file = os.path.join(self.args.save_dir, 'loss_curve.png')
            loss_file = os.path.join(self.args.save_dir, 'loss')
        plt.savefig(loss_curve_file)
        plt.clf()

        torch.save(self.loss_buffer, loss_file)


    def attack_loss(self, prompts, targets, images=None):

        context_embs = prompts.context_embs
        assert len(context_embs) == len(targets), "Unmathced batch size of prompts and targets, the length of context_embs is %d, the length of targets is %d" % (len(context_embs), len(targets))

        batch_size = len(targets)
        
        # Use provided images or fallback to self.safe_image
        if images is None:
            images = self.safe_image

        to_regress_tokens = self.tokenizer(
            targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=160,
            add_special_tokens=False
        ).to(self.device)
        to_regress_embs = self.model.model.embed_tokens(to_regress_tokens.input_ids)

        bos = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.tokenizer.bos_token_id
        bos_embs = self.model.model.embed_tokens(bos)

        pad = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.tokenizer.pad_token_id
        pad_embs = self.model.model.embed_tokens(pad)


        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )

        pos_padding = torch.argmin(T, dim=1) # a simple trick to find the start position of padding

        input_embs = []
        targets_mask = []

        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []

        for i in range(batch_size):

            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]

            targets_mask.append(T[i:i+1, :target_length])
            input_embs.append(to_regress_embs[i:i+1, :target_length]) # omit the padding tokens

            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length

            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)

        max_length = max(seq_tokens_length)

        attention_mask = []

        for i in range(batch_size):

            # masked out the context from loss computation
            context_mask =(
                torch.ones([1, context_tokens_length[i] + 1],
                       dtype=torch.long).to(self.device).fill_(-100)  # plus one for bos
            )

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = (
                torch.ones([1, num_to_pad],
                       dtype=torch.long).to(self.device).fill_(-100)
            )

            targets_mask[i] = torch.cat( [context_mask, targets_mask[i], padding_mask], dim=1 )
            input_embs[i] = torch.cat( [bos_embs, context_embs[i], input_embs[i],
                                        pad_embs.repeat(1, num_to_pad, 1)], dim=1 )
            attention_mask.append( torch.LongTensor( [[1]* (1+seq_tokens_length[i]) + [0]*num_to_pad ] ) )

        targets = torch.cat( targets_mask, dim=0 ).to(self.device)
        inputs_embs = torch.cat( input_embs, dim=0 ).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)

        # Prepare images for batch if provided
        if images is not None:
            # Ensure images have correct shape: [batch_size, C, H, W]
            if len(images.shape) == 3:  # [C, H, W]
                images = images.unsqueeze(0)  # [1, C, H, W]
            images = images.to(self.device)

        # Model forward pass - use full multimodal model if images provided
        if images is not None:
            # Use full model with images (multimodal)
            # Process in smaller chunks to save memory (multimodal model is memory-intensive)
            hidden_states_list = []
            # Get chunk_size from args if available, otherwise use default
            chunk_size = getattr(self.args, 'chunk_size', 2)  # Default to 2 if not specified
            for i in range(0, batch_size, chunk_size):
                chunk_end = min(i + chunk_size, batch_size)
                chunk_inputs_embs = inputs_embs[i:chunk_end]
                chunk_attention_mask = attention_mask[i:chunk_end]
                
                # Repeat image for this chunk (LLaVA requires image batch to match text batch)
                chunk_images = images.repeat(chunk_end - i, 1, 1, 1) if images.shape[0] == 1 else images[i:chunk_end]
                
                outputs = self.model(
                    input_ids=None,
                    inputs_embeds=chunk_inputs_embs,
                    attention_mask=chunk_attention_mask,
                    images=chunk_images.half(),
                    return_dict=True,
                    output_hidden_states=True,
                )
                # CausalLMOutputWithPast has hidden_states as a list, get the last one
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states_list.append(outputs.hidden_states[-1])
                else:
                    raise ValueError("Model output does not contain hidden_states despite output_hidden_states=True")
                
                # Clear cache after each chunk to free memory
                del outputs
                torch.cuda.empty_cache()
            
            hidden_states = torch.cat(hidden_states_list, dim=0)
        else:
            # Use text-only model (backward compatibility)
            outputs = self.model.model(
                inputs_embeds=inputs_embs,
                attention_mask=attention_mask,
            )
            hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        loss = None
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        # Flatten the tokens
        from torch.nn import CrossEntropyLoss
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.model.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model/pipeline parallelism - ensure both tensors are on the same device
        shift_logits = shift_logits.to(self.device)
        shift_labels = shift_labels.to(self.device)
        loss = loss_fct(shift_logits, shift_labels)

        return loss
