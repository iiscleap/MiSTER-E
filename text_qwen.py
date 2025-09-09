import logging
import json
import contextlib
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoModelForCausalLM
)
from peft import LoraConfig, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training
# from llama.models.modeling_llama import LlamaForCausalLM
import json
import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os
import json
import logging
import argparse


BATCH_SIZE = 32
LEARNING_RATE = 1e-5
BETAS = (0.9, 0.99)
EPS = 1e-06
WEIGHT_DECAY = 0.0
MAX_NORM = 10
STEPS = 800000
#Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

np.random.seed(42)
torch.manual_seed(42)

#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

class IEMOCAP6Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, labels, tokenizer):
        self.llama_tokenizer = tokenizer
        self.wavs = list(data_dict.keys())
        self.wavs = [x for x in self.wavs if x in labels]
        self.text_dict = data_dict
        self.label_dict = labels

    def __getitem__(self, index):
        wav_name = self.wavs[index]
        text_raw = self.text_dict[wav_name]
        emo_labels = self.label_dict[wav_name]

        return text_raw, emo_labels, wav_name

    def __len__(self):
        return len(self.wavs)
    
    def collate(self, batch):
        tokenized_text, emo_labels, wav_name = zip(*batch)
        tokenized_text = [t+ "</s>" for t in tokenized_text]
        tokenized_text = self.llama_tokenizer(tokenized_text, return_tensors="pt",padding="longest",truncation=True,max_length=128,add_special_tokens=False)

        return tokenized_text, torch.tensor(emo_labels), wav_name
    
class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
    def compute_length_from_mask(self, mask):
        """
        mask: (batch_size, T)
        Assuming that the sampling rate is 16kHz, the frame shift is 20ms
        """
        wav_lens = torch.sum(mask, dim=1) # (batch_size, )
        # feat_lens = torch.div(wav_lens-1, 16000*0.02, rounding_mode="floor") + 1
        feat_lens = wav_lens.int().tolist()
        return feat_lens
        
    def forward(self, x, mask):
        raise NotImplementedError
    

class AttentiveStatisticsPooling(Pooling):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    Link: https://arxiv.org/pdf/1803.10963.pdf
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs, mask):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)

        => output: (batch_size, feat_dim*2)
        """
        feat_lens = self.compute_length_from_mask(mask)
        pooled_list = []
        for x, feat_len in zip(xs, feat_lens):
            x = x[:feat_len].unsqueeze(0)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1).squeeze(0)
            pooled_list.append(x)
        return torch.stack(pooled_list)
    
class LLamaSentiment(nn.Module):
    def __init__(self,
                 llama,
                 tokenizer,
                 hidden_dim,
                 output_dim):
        
        super().__init__()
        
        self.llama = llama
        self.llama_tokenizer = tokenizer
        self.output_dim = 3584
        # self.weights = nn.Parameter(torch.rand(41, 1))
        self.out = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(self.output_dim*2, hidden_dim)
        # self.attenpool = SelfAttentionPooling(3584)
        self.pool = AttentiveStatisticsPooling(self.output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.ln = nn.LayerNorm(self.output_dim)

    def get_features(self, text):
        embeds = self.llama.model.model.embed_tokens(text.input_ids)
        inputs_embeds = torch.cat([embeds], dim=1)
        attention_mask = torch.cat([text.attention_mask], dim=1)
        outputs = self.llama.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
        
        feat = torch.stack(list(outputs['hidden_states']), dim=0)
        # feat = torch.sum(feat[-4:, :, :, :], 0) /4
        feat = feat[-1, :, :, :]
        feat = self.ln(feat)
        feat = self.pool(feat, text.attention_mask)
        feat = self.fc(feat)
        return feat.to(torch.float32)
        
        
    def forward(self, text):
        embeds = self.llama.model.model.embed_tokens(text.input_ids)
        inputs_embeds = torch.cat([embeds], dim=1)
        attention_mask = torch.cat([text.attention_mask], dim=1)
        outputs = self.llama.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
        
        feat = torch.stack(list(outputs['hidden_states']), dim=0)
        # feat = torch.sum(feat[-4:, :, :, :], 0) /4
        feat = feat[-1, :, :, :]
        feat = self.ln(feat)
        feat = self.pool(feat, text.attention_mask)
        
        output = self.out(self.dropout(self.relu(self.fc(feat))))
        return output

# class LLamaSentiment(nn.Module):
#     def __init__(self,
#                  llama,
#                  tokenizer,
#                  hidden_dim,
#                  output_dim):
        
#         super().__init__()
        
#         self.llama = llama
#         self.llama_tokenizer = tokenizer
#         self.output_dim = 3584
#         # self.weights = nn.Parameter(torch.rand(41, 1))
#         self.out = nn.Linear(hidden_dim, output_dim)
#         self.fc = nn.Linear(self.output_dim, hidden_dim)
#         self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
#         # self.attenpool = SelfAttentionPooling(3584)
#         self.pool = AttentiveStatisticsPooling(hidden_dim)

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.ln = nn.LayerNorm(hidden_dim)
#         self.ln1 = nn.LayerNorm(hidden_dim)
        
        
#     def forward(self, text):
#         embeds = self.llama.model.model.embed_tokens(text.input_ids)
#         inputs_embeds = torch.cat([embeds], dim=1)
#         attention_mask = torch.cat([text.attention_mask], dim=1)
#         outputs = self.llama.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 return_dict=True,
#             )
        
#         feat = torch.stack(list(outputs['hidden_states']), dim=0)
#         feat = feat[-1, :, :, :]
#         feat = self.fc(feat)
#         feat = self.ln(feat)
#         feat = self.dropout(self.relu(feat))
#         feat = self.pool(feat, text.attention_mask)

#         feat = self.fc1(feat)
#         feat = self.ln1(feat)
#         feat = self.relu(feat)
#         feat = self.dropout(feat)
#         output = self.out(feat)

        
#         # output = self.out(self.dropout(self.relu(self.fc(feat))))
#         return output


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma, weight=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  # Optional: can be used for class weighting

    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        softmax = torch.softmax(inputs, dim=1)
        # Get the probability of the true class for each sample
        p_t = softmax.gather(1, targets.unsqueeze(1))
        # Compute the focal loss part
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)

        # If weight is provided, apply it per class
        if self.weight is not None:
            weight = self.weight[targets]
            loss = loss * weight
        
        return loss.mean()

class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, pred, target):
        output = pred / self.Delta_list + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)

def create_dataset(mode, tokenizer, bs):
    f = open("/home/soumyadutta/HCAM/IEMOCAP6/text/transcripts.json")
    sent_dict = json.load(f)
    f.close()
    if mode == 'train':
        labels = open("/home/soumyadutta/HCAM/IEMOCAP6/labels/train_wav_labels.json")
        labels_dict = json.load(labels)
        labels.close()
    elif mode == 'val':
        labels = open("/home/soumyadutta/HCAM/IEMOCAP6/labels/val_wav_labels.json")
        labels_dict = json.load(labels)
        labels.close()
    else:
        labels = open("/home/soumyadutta/HCAM/IEMOCAP6/labels/test_wav_labels.json")
        labels_dict = json.load(labels)
        labels.close()
    dataset = IEMOCAP6Dataset(sent_dict, labels_dict,tokenizer)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader

def train():
    llama_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", use_fast=False)
    # llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"

    train_loader = create_dataset("train", llama_tokenizer, 8)
    val_loader = create_dataset("val", llama_tokenizer, 1)
    num_classes = 6
    class_weights = torch.tensor([1.1842, 1.2867, 0.303, 0.8187, 2.0893, 5.3246, 5.2657]).to(device)
    class_numbers = np.array([1205, 1109, 4710, 1743, 683, 268, 271])
    alpha = 1.0
    gamma = 2.0
    criterion = WeightedFocalLoss(alpha=alpha, gamma=gamma)
    # criterion = VSLoss(class_numbers, weight=class_weights)
    
    llama_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B",torch_dtype=torch.float16,device_map={"":0},output_hidden_states=True, num_labels=8)
    llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
    llama_model.resize_token_embeddings(len(llama_tokenizer))
    # for name, param in llama_model.named_parameters():
    #     param.requires_grad = False
    logging.info('Loading LLaMA Done')
    peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.1,
            )

    # peft_config = LoraConfig(
    #     r = 16, # the dimension of the low-rank matrices
    #     lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    #     # target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    #     lora_dropout = 0.05, # dropout probability of the LoRA layers
    #     bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    #     task_type = 'SEQ_CLS'
    # )
    
    llama_model = get_peft_model(llama_model, peft_config)
    llama_model.print_trainable_parameters()
    logging.info('LoRA Training')

    model = LLamaSentiment(llama_model, llama_tokenizer, 2048, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
            params,
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
        )

    final_val_loss = 0
    accumulation_steps = 1
    for e in range(20):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        for i, data in enumerate(tqdm(train_loader)):
            text, labels, _ = data
            labels = labels.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                final_out = model(text.to(device))
            # loss = compute_loss(final_out, labels)
            loss = criterion(final_out, labels)
            loss = loss / accumulation_steps
            loss.backward()
            tot_loss += loss.detach().item()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            pred = torch.argmax(final_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                text, labels, _ = data
                labels = labels.to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    val_out = model(text.to(device))
                loss = criterion(val_out, labels)
                val_loss += loss.item()
                #val_correct += correct.item()
                pred = torch.argmax(val_out, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)
        val_f1 = f1_score(gt_val, pred_val, average='weighted')
        if val_f1 > final_val_loss:
            state_dict = model.state_dict()
            param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
            }
            for k in list(state_dict.keys()):
                if k in param_grad_dic.keys() and not param_grad_dic[k]:
                    del state_dict[k]
            torch.save(state_dict, "iemocap6.pth")
            final_val_loss = val_f1
        train_loss = tot_loss/len(train_loader)
        train_f1 = f1_score(gt_tr, pred_tr, average='weighted')
        val_loss_log = val_loss/len(val_loader)
        val_f1 = f1_score(gt_val, pred_val, average='weighted')
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}")
        
# def test():
#     num_classes = 8
#     base_model = RobertaForSequenceClassification.from_pretrained('roberta-large', output_hidden_states = True)
#     model = RobertaGRUSentiment(base_model, 384, num_classes, 2, True, 0.1)
#     model = torch.load("roberta_large.pth")
#     model.to(device)
#     model.eval()
#     f = open("/data1/soumyad/IS2025_challenge/test_transcripts.json")
#     details = json.load(f)
#     f.close()
#     files = details["path"]
#     transcripts = details["text"]
#     text_dict = dict(zip(files, transcripts))
#     tokenizer = AutoTokenizer.from_pretrained("roberta-base")
#     label_map = {0:"A", 1:"C", 2:"D", 3:"F", 4:"H", 5:"N", 6:"S", 7:"U"}
#     predicted_dict = {"FileName":[], "EmoClass":[]}
#     with torch.no_grad():
#         for f in text_dict:
#             wav_name = f.split(os.sep)[-1]
#             text = text_dict[f]
#             text = tokenizer(text, padding=True, return_tensors="pt", return_special_tokens_mask=False)
#             final_out = model(text['input_ids'].to(device), attention_mask=text['attention_mask'].to(device))[0]
#             pred = torch.argmax(final_out, dim = 1)
#             pred = pred.detach().cpu().numpy()[0]
#             predicted_dict["FileName"].append(wav_name)
#             predicted_dict["EmoClass"].append(label_map[pred])
#     df = pd.DataFrame(predicted_dict)
#     df.to_csv("categorical_text.csv", index=False)

def test():


    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=False)
    # llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"
    val_loader = create_dataset("test", llama_tokenizer, 1)
    num_classes = 6

    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B",torch_dtype=torch.float16,device_map={"":0},output_hidden_states=True, num_labels=num_classes)
    llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
    llama_model.resize_token_embeddings(len(llama_tokenizer))

    peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.1,
            )

    llama_model = get_peft_model(llama_model, peft_config)
    llama_model.print_trainable_parameters()
    logging.info('LoRA Training')

    model = LLamaSentiment(llama_model, llama_tokenizer, 2048, num_classes)
    model.to(device)
    model.load_state_dict(torch.load("iemocap6.pth", weights_only=True), strict=False)
    model.to(device)
    model.eval()
    pred_test, gt_test = [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
            text, labels, _ = data
            labels = labels.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                val_out = model(text.to(device))
            pred = torch.argmax(val_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_test.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_test.extend(labels)
    test_f1 = f1_score(gt_test, pred_test, average='weighted')
    logger.info(f"Test Accuracy {test_f1}")


def get_features():


    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=False)
    # llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"
    data_loader = create_dataset("val", llama_tokenizer, 1)
    num_classes = 7

    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B",torch_dtype=torch.float16,device_map={"":0},output_hidden_states=True, num_labels=num_classes)
    llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
    llama_model.resize_token_embeddings(len(llama_tokenizer))

    peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.1,
            )

    llama_model = get_peft_model(llama_model, peft_config)
    llama_model.print_trainable_parameters()
    logging.info('LoRA Training')

    model = LLamaSentiment(llama_model, llama_tokenizer, 2048, num_classes)
    model.to(device)
    model.load_state_dict(torch.load("test.pth", weights_only=True), strict=False)
    model.to(device)
    model.eval()
    os.makedirs(os.path.join("/home/soumyadutta/HCAM/MELD/text", "text_feats", "val"), exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            text, labels, wav_name = data
            labels = labels.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                val_out = model.get_features(text.to(device))
            val_out = val_out.squeeze(0)
            val_out = val_out.detach().cpu().numpy()
            out_file = os.path.join("/home/soumyadutta/HCAM/MELD/text", "text_feats", "val", wav_name[0].replace(".wav", ".npy"))
            np.save(out_file, val_out)
    #         pred = torch.argmax(val_out, dim = 1)
    #         pred = pred.detach().cpu().numpy()
    #         pred = list(pred)
    #         pred_test.extend(pred)
    #         labels = labels.detach().cpu().numpy()
    #         labels = list(labels)
    #         gt_test.extend(labels)
    # test_f1 = f1_score(gt_test, pred_test, average='weighted')
    # logger.info(f"Test Accuracy {test_f1}")

# def test():


#     llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=False)
#     # llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     llama_tokenizer.pad_token = llama_tokenizer.eos_token
#     llama_tokenizer.padding_side = "right"
#     # val_loader = create_dataset("val", llama_tokenizer, 1)

#     f = open("test_transcripts.json")
#     details = json.load(f)
#     f.close()
#     files = details["path"]
#     transcripts = details["text"]
#     text_dict = dict(zip(files, transcripts))
#     num_classes = 8

#     llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B",torch_dtype=torch.float16,device_map={"":0},output_hidden_states=True, num_labels=num_classes)
#     llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
#     llama_model.resize_token_embeddings(len(llama_tokenizer))

#     peft_config = LoraConfig(
#                 task_type=TaskType.CAUSAL_LM, 
#                 inference_mode=False, 
#                 r=8, 
#                 lora_alpha=32, 
#                 lora_dropout=0.1,
#             )

#     llama_model = get_peft_model(llama_model, peft_config)
#     llama_model.print_trainable_parameters()
#     logging.info('LoRA Training')

#     model = LLamaSentiment(llama_model, llama_tokenizer, 1024, num_classes)
#     model.to(device)
#     model.load_state_dict(torch.load("llama8b.pth", weights_only=True), strict=False)
#     model.to(device)
#     model.eval()
#     label_map = {0:"A", 1:"C", 2:"D", 3:"F", 4:"H", 5:"N", 6:"S", 7:"U"}
#     predicted_dict = {"FileName":[], "EmoClass":[]}


#     with torch.no_grad():
#         for i, f in enumerate(tqdm(files)):
#             wav_name = f.split(os.sep)[-1]
#             text = text_dict[f]
#             tokenized_text = text+ "</s>"
#             tokenized_text = llama_tokenizer(tokenized_text, return_tensors="pt",padding="longest",truncation=True,max_length=128,add_special_tokens=False)

#             with torch.cuda.amp.autocast(enabled=True):
#                 val_out = model(tokenized_text.to(device))
#             pred = torch.argmax(val_out, dim = 1)
#             pred = pred.detach().cpu().numpy()[0]
#             predicted_dict["FileName"].append(wav_name)
#             predicted_dict["EmoClass"].append(label_map[pred])

#     df = pd.DataFrame(predicted_dict)
#     df.to_csv("llama_test.csv", index=False)


if __name__ == "__main__":
    train()
    test()
    # get_features()
        
