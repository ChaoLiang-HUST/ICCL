# coding: UTF-8

import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
embedding_size = 768


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.RoBERTa_MLM = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
        self.RoBERTa_MLM.resize_token_embeddings(args.vocab_size)
        for param in self.RoBERTa_MLM.parameters():
            param.requires_grad = True

        self.hidden_size = 768

        self.vocab_size = args.vocab_size
        self.temperature = args.temperature

    def forward(self, arg, mask_arg, token_mask_indices, position, Answer_id, mode='Contrastive Learning'):
        # Test and Dev mode
        if mode == 'Prompt Learning':
            out_arg = self.RoBERTa_MLM.roberta(arg, attention_mask=mask_arg, output_hidden_states=True)[0].cuda()
            anchor_hidden_mask = torch.zeros((len(arg), self.hidden_size)).cuda()
            for i in range(len(arg)):
                anchor_hidden_mask[i] = out_arg[i][token_mask_indices[i]]
            out_vocab = self.RoBERTa_MLM.lm_head(anchor_hidden_mask)
            out_ans = out_vocab[:, Answer_id]
            return out_ans
        # Training mode
        elif mode == 'Contrastive Learning':
            pos_num = len(position[0][0])
            neg_num = len(position[0][1])
            all_hidden = self.RoBERTa_MLM.roberta(arg, attention_mask=mask_arg, output_hidden_states=True)[0].cuda()

            # Contrastive loss calculation
            all_hidden_mask = torch.zeros((len(all_hidden), self.hidden_size)).cuda()
            anchor_hidden_mask = torch.zeros((len(all_hidden), self.hidden_size)).cuda()
            pos_hidden_mask = torch.zeros((len(all_hidden), pos_num, self.hidden_size)).cuda()
            neg_hidden_mask = torch.zeros((len(all_hidden), neg_num, self.hidden_size)).cuda()
            for i in range(len(all_hidden)):
                all_hidden_mask[i] = all_hidden[i][token_mask_indices[i]]
                a = token_mask_indices[i].item()
                anchor_hidden_mask[i] = all_hidden[i][a - 1] - all_hidden[i][a + 1]
                for j in range(pos_num):
                    p = position[i][0][j]
                    pos_hidden_mask[i][j] = all_hidden[i][p - 1] - all_hidden[i][p + 1]
                for j in range(neg_num):
                    n = position[i][1][j]
                    neg_hidden_mask[i][j] = all_hidden[i][n - 1] - all_hidden[i][n + 1]

            anc_norm = torch.sqrt(torch.sum(anchor_hidden_mask * anchor_hidden_mask, dim=1))
            pos_norm = torch.sqrt(torch.sum(pos_hidden_mask * pos_hidden_mask, dim=2))
            neg_norm = torch.sqrt(torch.sum(neg_hidden_mask * neg_hidden_mask, dim=2))

            pos_sim = torch.zeros((len(all_hidden), 1)).cuda()
            neg_sim = torch.zeros((len(all_hidden), 1)).cuda()
            for i in range(len(all_hidden)):
                pos_sim[i] = torch.sum(torch.exp((torch.mm(pos_hidden_mask[i],  anchor_hidden_mask[i].unsqueeze(dim=1)) /
                                                  pos_norm[i].unsqueeze(dim=1) / anc_norm[i]) / self.temperature),
                                       dim=0)
                neg_sim[i] = torch.sum(torch.exp((torch.mm(neg_hidden_mask[i],  anchor_hidden_mask[i].unsqueeze(dim=1)) /
                                                  neg_norm[i].unsqueeze(dim=1) / anc_norm[i]) / self.temperature),
                                       dim=0)

            CL_loss = - torch.log(pos_sim / (pos_sim + neg_sim))
            CL_loss = torch.sum(CL_loss, dim=0) / len(all_hidden)

            # Prediction
            out_vocab = self.RoBERTa_MLM.lm_head(all_hidden_mask)
            out_ans = out_vocab[:, Answer_id]
            return CL_loss, out_ans

    # Initialize the embeddings of special tokens we added
    def handler(self, to_add, tokenizer):
        da = self.RoBERTa_MLM.roberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp
