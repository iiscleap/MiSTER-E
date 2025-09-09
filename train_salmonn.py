import os
import torch
import torchaudio
import librosa
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam, AdamW
import torch.nn as nn
import random
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import random
import argparse
import time
from argparse import Namespace
import pandas as pd
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import torch.nn.functional as F
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training

#Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Fine tune")
parser.add_argument(
    "--seed",
    metavar="seed",
    type=int,
)


args = parser.parse_args()

SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

class SpeechDataset(Dataset):
    def __init__(
        self,
        folder,
        labels
    ):
        self.folder = folder
        self.files = os.listdir(folder)
        self.label_dict = labels
        self.files = [x for x in self.files if ".wav" in x]
        self.wav_files = [x for x in self.files if x in labels]
        self.sr = 16000
        self.duration = 5000
        self.prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>What is the emotion of the person?:"
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)

    def __len__(self):
        return len(self.wav_files)


    def __getitem__(self, index):
        wav_name = self.wav_files[index]
        label = self.label_dict[wav_name]

        audio_file = os.path.join(self.folder, wav_name)   
        (sig, sr) = librosa.load(audio_file, sr=None)

        aud = (sig, sr)
        reaud = (sig, self.sr)
        resig = sig
        sig_len = resig.shape[0]
        max_len = self.sr//1000 * self.duration
        if len(resig.shape) == 2:
            resig = np.mean(resig, axis = 1)

        if (sig_len > max_len):
            # Truncate the signal to the given length
            start = np.random.randint(0, sig_len-max_len)

            final_sig = resig[start:start+max_len]

        elif sig_len < max_len:
            pad_end_len = max_len - sig_len

            # # Pad with 0s
            # pad_begin = np.zeros((pad_begin_len))
            pad_end = np.ones((pad_end_len))*1e-6

            final_sig = np.float64(np.concatenate((resig, pad_end), 0))
        else:
            final_sig = resig
        
        inputs = self.processor(text=self.prompt, audio=final_sig, sampling_rate=sr, return_tensors="pt")
        return {
            "inputs": inputs,
            "labels": label,
            "wav_name": wav_name.split(os.sep)[-1]
        }

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
        feat_lens = torch.div(wav_lens-1, 16000*0.02, rounding_mode="floor") + 1
        feat_lens = feat_lens.int().tolist()
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
        # feat_lens = self.compute_length_from_mask(mask)
        pooled_list = []
        for x in xs:
            x = x.unsqueeze(0)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1).squeeze(0)
            pooled_list.append(x)
        return torch.stack(pooled_list)



class EmotionClassifier(nn.Module):
    def __init__(self,
                 qwen_model,
                 hidden_dim,
                 output_dim):
        
        super().__init__()
        self.qwen_model = qwen_model
        self.out = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(4096, hidden_dim)
        self.ln = nn.LayerNorm(4096)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def get_features(self, input_ids, att_mask, input_features, feature_attention_mask):

        feats_final = self.qwen_model(input_ids=input_ids.squeeze(1), attention_mask=att_mask.squeeze(1), input_features=input_features.squeeze(1), feature_attention_mask=feature_attention_mask.squeeze(1))
        feats_final = feats_final["hidden_states"][-1]
        feats_final = self.ln(feats_final)
        feat = torch.mean(feats_final, 1)
        feat = self.fc(feat)

        return feat.to(torch.float32)

        
    def forward(self, input_ids, att_mask, input_features, feature_attention_mask):
        feats_final = self.qwen_model(input_ids=input_ids.squeeze(1), attention_mask=att_mask.squeeze(1), input_features=input_features.squeeze(1), feature_attention_mask=feature_attention_mask.squeeze(1))
        feats_final = feats_final["hidden_states"][-1]
        feats_final = self.ln(feats_final)
        feat = torch.mean(feats_final, 1)
        feat = self.dropout(self.relu(self.fc(feat)))
        output = self.out(feat)
        
        return output

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

def compute_accuracy(output, labels):
    #Function for calculating accuracy
    pred = torch.argmax(output, dim = 1)
    correct_pred = (pred == labels).float()
    tot_correct = correct_pred.sum()

    return tot_correct

def compute_loss(output, labels):
    #Function for calculating loss

    ce_loss = nn.CrossEntropyLoss(reduction='none')(output, labels.squeeze(-1).long())
    pt = torch.exp(-ce_loss)
    loss = ((1-pt)**0 * ce_loss).mean()
    return loss

def create_dataset(mode, bs=8):
    folder = "/home/soumyadutta/HCAM/IEMOCAP6/audio/wavs"
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
    dataset = SpeechDataset(folder, labels_dict)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=True)
    return loader
    
def train():

    train_loader = create_dataset("train", 2)
    val_loader = create_dataset("val", 1)
    num_classes = 6
    class_weights = torch.tensor([1.1842, 1.2867, 0.303, 0.8187, 2.0893, 5.3246, 5.2657]).to(device)
    class_numbers = np.array([1205, 1109, 4710, 1743, 683, 268, 271])

    alpha = 1.0
    gamma = 2.0
    criterion = WeightedFocalLoss(alpha=alpha, gamma=gamma)
    
    qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True, output_hidden_states=True)
    peft_config = LoraConfig(
        r = 16, # the dimension of the low-rank matrices
        lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
        target_modules = ['q_proj', 'k_proj'],
        lora_dropout = 0.05, # dropout probability of the LoRA layers
        bias = 'none', # wether to train bias weights, set to 'none' for attention layers
        task_type = 'SEQ_CLS'
    )
    qwen_model = get_peft_model(qwen_model, peft_config)
    qwen_model.print_trainable_parameters()
    logging.info('LoRA Training')
    model = EmotionClassifier(qwen_model, 2048, num_classes)
    model.to(device)
    for name, param in model.named_parameters():
        if "audio_tower" in name:
            param.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    base_lr = 1e-5
    optimizer = AdamW(
            params,
            lr=base_lr
        )
    final_val_loss = 0
    accumulation_steps = 8
    scaler = torch.cuda.amp.GradScaler() 
    for e in range(100000):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        val_size = 0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        for i, data in enumerate(tqdm(train_loader)):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data["inputs"].to(device), data["labels"].to(device)
            with torch.amp.autocast('cuda', enabled=True):
                final_out = model(inputs["input_ids"], inputs["attention_mask"], inputs["input_features"], inputs["feature_attention_mask"])
                loss = criterion(final_out, labels)/ accumulation_steps
                loss = loss / accumulation_steps

            scaler.scale(loss).backward() 
            tot_loss += loss.detach().item()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
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
                inputs, labels = data["inputs"].to(device), data["labels"].to(device)    #             samples = {"spectrogram": spectrogram,"raw_wav": aud,"padding_mask": padding_mask}
                with torch.amp.autocast('cuda', enabled=True):
                    val_out = model(inputs["input_ids"], inputs["attention_mask"], inputs["input_features"], inputs["feature_attention_mask"])
                    loss = criterion(val_out, labels) / accumulation_steps
                val_loss += loss.item()
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
            # torch.save(state_dict, "qwen_iemocap6.pth")
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
        
def test():

    val_loader = create_dataset("test", 1)
    num_classes = 6
    qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True, output_hidden_states=True)
    peft_config = LoraConfig(
        r = 16, # the dimension of the low-rank matrices
        lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
        target_modules = ['q_proj', 'k_proj'],
        lora_dropout = 0.05, # dropout probability of the LoRA layers
        bias = 'none', # wether to train bias weights, set to 'none' for attention layers
        task_type = 'SEQ_CLS'
    )
    qwen_model = get_peft_model(qwen_model, peft_config)
    qwen_model.print_trainable_parameters()
    model = EmotionClassifier(qwen_model, 2048, num_classes)
    model.to(device)
    model.load_state_dict(torch.load("qwen_iemocap6.pth", weights_only=True), strict=False)
    model.to(device)
    model.eval()

    pred_test, gt_test = [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
            inputs, labels = data["inputs"].to(device), data["labels"].to(device)
            with torch.amp.autocast('cuda', enabled=True):
                test_out = model(inputs["input_ids"], inputs["attention_mask"], inputs["input_features"], inputs["feature_attention_mask"])
            pred = torch.argmax(test_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_test.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_test.extend(labels)
    test_f1 = f1_score(gt_test, pred_test, average='weighted')
    logger.info(f"Test Accuracy {test_f1}")

# def test():
#     test_loader = create_dataset("val", 1)
#     num_classes = 8
#     salmonn_model = SALMONN.from_config(cfg.config.model)
#     model = EmotionClassifier(salmonn_model, 1024, num_classes)
#     model.load_state_dict(torch.load("salmonn_podcast_drop.pth"), strict=False)
#     model.to(device)
#     model.eval()
#     f = open("/data1/soumyad/IS2025_challenge/test_transcripts.json")
#     details = json.load(f)
#     f.close()
#     files = details["path"]
#     label_map = {0:"A", 1:"C", 2:"D", 3:"F", 4:"H", 5:"N", 6:"S", 7:"U"}
#     predicted_dict = {"FileName":[], "EmoClass":[]}
    
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(test_loader)):
#             aud, spectrogram, target, padding_mask = data["raw_wav"].to(device), data["spectrogram"].to(device), data["labels"].to(device), data["padding_mask"].to(device)
#             samples = {"spectrogram": spectrogram,"raw_wav": aud,"padding_mask": padding_mask}
#             wav_name = data["wav_name"][0]
#             with torch.cuda.amp.autocast(enabled=True):
#                 test_out = model(samples, aud)
#             pred = torch.argmax(test_out, dim = 1)
#             pred = pred.detach().cpu().numpy()[0]
#             predicted_dict["FileName"].append(wav_name)
#             predicted_dict["EmoClass"].append(label_map[pred])
#     df = pd.DataFrame(predicted_dict)
#     df.to_csv("salmonn_drop_valid.csv", index=False)


def get_features():
    num_classes = 6
    qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True, output_hidden_states=True)
    peft_config = LoraConfig(
        r = 16, # the dimension of the low-rank matrices
        lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
        target_modules = ['q_proj', 'k_proj'],
        lora_dropout = 0.05, # dropout probability of the LoRA layers
        bias = 'none', # wether to train bias weights, set to 'none' for attention layers
        task_type = 'SEQ_CLS'
    )
    qwen_model = get_peft_model(qwen_model, peft_config)
    qwen_model.print_trainable_parameters()
    model = EmotionClassifier(qwen_model, 2048, num_classes)
    model.to(device)
    model.load_state_dict(torch.load("qwen_iemocap6.pth", weights_only=True), strict=False)
    model.to(device)
    model.eval()

    labels = open("/home/soumyadutta/HCAM/IEMOCAP6/labels/test_wav_labels.json")
    labels_dict = json.load(labels)
    labels.close()

    folder = "/home/soumyadutta/HCAM/IEMOCAP6/audio/wavs"
    wav_files = os.listdir(folder)
    wav_files = [x for x in wav_files if ".wav" in x]
    wav_files = [x for x in wav_files if x in labels_dict]
    output_folder = "qwen_audio_feats"
    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        for i, wav_name in enumerate(tqdm(wav_files)):
            f = os.path.join(folder, wav_name)
            out_file = os.path.join(output_folder, f.split(os.sep)[-1].replace(".wav", ".npy"))
            # if os.path.exists(out_file):
            #     continue
            audio, sr = librosa.load(f, sr=None)
            sr = 16000
            if len(audio.shape) == 2: # stereo to mono
                audio = audio[:, 0]
            if len(audio) < sr: # pad audio to at least 1s
                sil = np.zeros(sr - len(audio), dtype=float)
                audio = np.concatenate((audio, sil), axis=0)
            audio = audio[: sr * 30]
            prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>What is the emotion of the person?:"
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True)
            inputs = processor(text=prompt, audio=audio, sampling_rate=sr, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                feat = model.get_features(inputs["input_ids"], inputs["attention_mask"], inputs["input_features"], inputs["feature_attention_mask"])
            # print(hidden_states.shape)
            np.save(out_file, feat.cpu().detach().numpy())



if __name__ == "__main__":
    train()
    test()
    # get_features()
