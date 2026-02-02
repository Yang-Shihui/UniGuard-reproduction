import torch
from llava.conversation import conv_llava_llama_2
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token


def prepare_text_prompt(user_prompt):

    qs = DEFAULT_IMAGE_TOKEN + '\n'+ user_prompt

    conv = conv_llava_llama_2.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt

# support batch implementation
class Prompt:
    # tokenization
    # turn to embeddings

    # padding? wait until targets have been appended
    # prepare labels? need to wait for targets

    def __init__(self, model, tokenizer, text_prompts=None, device='cuda:0',max_new_tokens=300, max_length=2000):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.text_prompts = text_prompts
        self.img_prompts = [[]]

        self.context_length = []
        self.input_ids = []
        self.do_tokenization(self.text_prompts)

        self.max_new_tokens = max_new_tokens
        self.max_length = max_length

        self.text_embs = self.generate_text_embedding(self.text_prompts)
        self.img_embs = [[]]
        self.update_context_embs()


    def do_tokenization(self, text_prompts):

        if text_prompts is None:
            self.input_ids = []
            self.context_length = []
            return
        if type(text_prompts) is list:
            text_prompts = text_prompts[0]

        input_ids = tokenizer_image_token(text_prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        self.input_ids = [input_ids]
        self.context_length = [input_ids.shape[1]]

    def update_context_embs(self):

        if len(self.text_embs) == len(self.img_embs):
            self.context_embs = self.generate_context_embedding(
                    self.text_embs, self.img_embs
                )
        else:
            self.context_embs = []

    def update_text_prompt(self, text_prompts):
        self.text_prompts = text_prompts
        self.text_embs = self.generate_text_embedding(self.text_prompts)
        self.update_context_embs()

    def generate_text_embedding(self, text_prompts):

        if text_prompts is None:
            return []

        text_embs = []
        for item in text_prompts: # for each prompt within a batch
            prompt_segs = item.split('<image>')  # each <ImageHere> corresponds to one image
            seg_tokens = [
                self.tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            embs = [self.model.model.embed_tokens(seg_t) for seg_t in seg_tokens] # text to embeddings
            text_embs.append(embs)

        return text_embs

    def generate_context_embedding(self, batch_text_embs, batch_img_embs):
        #assert len(text_embs) == len(img_embs) + 1, "Unmatched numbers of image placeholders and images."

        assert len(batch_text_embs) == len(batch_img_embs), "Unmathced batch size of text and image prompts"

        batch_size = len(batch_text_embs)
        batch_context_embs = []

        for i in range(batch_size):

            mixed_embs = torch.cat(batch_text_embs[i], dim=1)
            current_max_len = mixed_embs.shape[1] + self.max_new_tokens
            if current_max_len - self.max_length > 0:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                      'The model will not see the contexts outside the range.')
            begin_idx = max(0, current_max_len - self.max_length)
            mixed_embs = mixed_embs[:, begin_idx:]
            
            batch_context_embs.append(mixed_embs)

        return batch_context_embs
