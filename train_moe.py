import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import json
from torch.optim import Adam, SGD
from sklearn.metrics import f1_score, classification_report
import argparse
import pickle
import csv
from hierarchical_moe import AttentionMoE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
    default=0
)
parser.add_argument(
        "--flag",
        metavar="flag",
        default="train",
        type=str,
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


class SequenceDataset(Dataset):
    def __init__(
        self,
        audio_seq,
        features_folder_aud,
        features_folder_text,
        label_seq
    ):
        self.audio_seq = audio_seq
        self.label_seq = label_seq
        self.features_file_aud = {}
        self.features_file_text = {}
        for i, (k, v) in enumerate(tqdm(self.audio_seq.items())):
            for utt in v:
                self.features_file_aud[utt] = np.load(os.path.join(features_folder_aud, utt.replace(".wav", ".npy")))
                self.features_file_text[utt] = np.load(os.path.join(features_folder_text, utt.replace(".wav", ".npy")))

    def __len__(self):
        return len(self.audio_seq)

    def __getitem__(self, index):

        conv_name = list(self.audio_seq.keys())[index]
        utterances = self.audio_seq[conv_name]
        labels_conversation = self.label_seq[conv_name]
        conversation_features_aud = []
        conversation_features_text = []
        conversation_labels = []
        for i, utt in enumerate(utterances):
            conversation_features_aud.append(torch.tensor(self.features_file_aud[utt]).reshape(2048,))
            conversation_features_text.append(torch.tensor(self.features_file_text[utt]))
            conversation_labels.append(torch.tensor(labels_conversation[i]))
        conversation_features_aud = torch.stack(conversation_features_aud, dim=0)
        conversation_features_text = torch.stack(conversation_features_text, dim=0)
        conversation_labels = torch.stack(conversation_labels, dim=0)
        

        return conversation_features_aud, conversation_features_text, conversation_labels, conv_name

    def collate(self, batch):
        conversations_feats_aud, conversations_feats_text, conversations_labels, conv_names = zip(*batch)

        conversations_feats_aud = list(conversations_feats_aud)
        conversations_feats_text = list(conversations_feats_text)
        conversations_labels = list(conversations_labels)
        conversations_lengths = [conversation_feat.size(0) for conversation_feat in conversations_feats_aud]

        max_conversation_length = max(conversations_lengths)


        collated_conversation_aud, collated_conversation_text, collated_labels = [], [], []

        for feat_aud, feat_text, label in zip(conversations_feats_aud, conversations_feats_text, conversations_labels):
            if len(feat_aud) < max_conversation_length:
                conv_diff = -feat_aud.size(0) + max_conversation_length
                padded_conv = torch.zeros((conv_diff, 2048))
                conv_aud = torch.cat((feat_aud, padded_conv), dim=0)
                conv_text = torch.cat((feat_text, padded_conv), dim=0)
                collated_conversation_aud.append(conv_aud)
                collated_conversation_text.append(conv_text)
                padded_label = torch.ones((conv_diff))*-1
                conv_label = torch.cat((label, padded_label), dim=0)
                collated_labels.append(conv_label)
            else:
                collated_conversation_aud.append(feat_aud)
                collated_conversation_text.append(feat_text)
                collated_labels.append(label)

        conv_features_aud = torch.stack(collated_conversation_aud, dim=0)
        conv_features_text = torch.stack(collated_conversation_text, dim=0)
        conv_labels = torch.stack(collated_labels, dim=0)


        return conv_features_text, conv_features_aud, conv_labels, conv_names


def create_IEMOCAP6_dataset(mode, bs=4):
    audio_features_folder = "/home/soumyadutta/HCAM/IEMOCAP6/audio/audio_feats"
    text_features_folder = "/home/soumyadutta/HCAM/IEMOCAP6/text/text_feats"
    if mode == 'train':
        seq_file= open("/home/soumyadutta/HCAM/IEMOCAP6/train_audio_sequence.json")
        audio_seq = json.load(seq_file)
        seq_file.close()
        label_file= open("/home/soumyadutta/HCAM/IEMOCAP6/labels/train_label_sequence.json")
        label_seq = json.load(label_file)
        label_file.close()
    elif mode == 'val':
        seq_file= open("/home/soumyadutta/HCAM/IEMOCAP6/val_audio_sequence.json")
        audio_seq = json.load(seq_file)
        seq_file.close()
        label_file= open("/home/soumyadutta/HCAM/IEMOCAP6/labels/val_label_sequence.json")
        label_seq = json.load(label_file)
        label_file.close()
    else:
        seq_file= open("/home/soumyadutta/HCAM/IEMOCAP6/test_audio_sequence.json")
        audio_seq = json.load(seq_file)
        seq_file.close()
        label_file= open("/home/soumyadutta/HCAM/IEMOCAP6/labels/test_label_sequence.json")
        label_seq = json.load(label_file)
        label_file.close()

    dataset = SequenceDataset(audio_seq, audio_features_folder, text_features_folder, label_seq)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=dataset.collate)
    return loader

def compute_accuracy(output, labels):
    pred = torch.argmax(output, -1)
    pred = pred.detach().cpu().numpy().reshape(-1)
    pred = list(pred)
    labels = labels.detach().cpu().numpy().reshape(-1)
    labels = list(labels)

    return pred, labels

def compute_loss(output, labels, num_classes):
    output = output.reshape(-1, num_classes)
    labels = labels.reshape(-1)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')(output, labels.long().to(device))
    pt = torch.exp(-ce_loss)
    loss = ((1-pt)**3* ce_loss).mean()

    return loss

def compute_supcon_loss(audio_feats, text_feats, labels, temperature=1):
    """
    Compute supervised contrastive loss at the time-step level.

    Args:
        audio_feats: Tensor of shape [B, T, D]
        text_feats: Tensor of shape [B, T, D]
        labels: Tensor of shape [B, T] (each timestep has a class label)
        temperature: Scaling factor for similarity

    Returns:
        contrastive_loss: scalar tensor
    """
    B, T, D = audio_feats.shape
    N = B * T

    # Flatten across time
    audio_flat = audio_feats.reshape(N, D)
    text_flat = text_feats.reshape(N, D)
    labels_flat = labels.reshape(N)

    # Normalize embeddings
    audio_flat = F.normalize(audio_flat, dim=-1)
    text_flat = F.normalize(text_flat, dim=-1)

    # Combine both modalities as two "views"
    all_reps = torch.cat([audio_flat, text_flat], dim=0)  # [2N, D]
    all_labels = torch.cat([labels_flat, labels_flat], dim=0)  # [2N]

    # Compute pairwise similarity
    sim_matrix = torch.matmul(all_reps, all_reps.T) / temperature  # [2N, 2N]

    # Mask: only consider different views of same class
    labels_matrix = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)  # [2N, 2N]
    logits_mask = ~torch.eye(2 * N, dtype=torch.bool, device=labels.device)  # exclude self-similarity
    positives_mask = labels_matrix & logits_mask

    # Log-softmax computation
    exp_sim = torch.exp(sim_matrix) * logits_mask.float()
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # Final contrastive loss: average over valid positives
    mean_log_prob_pos = (log_prob * positives_mask.float()).sum(1) / (positives_mask.sum(1) + 1e-12)
    loss = -mean_log_prob_pos.mean()

    return loss

def compute_supcon_loss_uni(feats, labels, temperature=0.07, device='cuda'):
    """
    Compute supervised contrastive loss for a batch of (B, T, D) features and (B, T) labels.
    Assumes -1 is the padding label and should be ignored.
    """
    eps = 1e-8

    # Flatten batch
    feats_flat = feats.reshape(-1, feats.size(-1))  # (B*T, D)
    labels_flat = labels.reshape(-1)                # (B*T,)

    # Filter out padded positions
    valid_mask = (labels_flat != -1)
    feats_valid = F.normalize(feats_flat[valid_mask], dim=1)  # (N, D)
    labels_valid = labels_flat[valid_mask]                    # (N,)

    N = feats_valid.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=feats.device, requires_grad=True)

    # Build similarity matrix
    sim_matrix = torch.exp(torch.matmul(feats_valid, feats_valid.T) / temperature)

    # Label mask
    labels_row = labels_valid.unsqueeze(1)  # (N, 1)
    labels_col = labels_valid.unsqueeze(0)  # (1, N)
    mask = (labels_row == labels_col).int().to(feats.device)  # (N, N)

    # Avoid self-comparison
    mask.fill_diagonal_(0)

    # Compute positives and negatives
    negatives = sim_matrix * (1 - mask)
    negative_sum = torch.sum(negatives, dim=1, keepdim=True) + eps

    # Log-ratio of positives to negatives
    log_ratios = torch.log(sim_matrix / negative_sum + eps) * mask

    # Average over positive pairs per anchor
    num_positives = torch.sum(mask, dim=1)
    num_positives = torch.clamp(num_positives, min=1)
    loss_per_sample = torch.sum(log_ratios, dim=1) / num_positives

    # Final loss
    sup_con_loss = -torch.mean(loss_per_sample)
    return sup_con_loss

def train():
    train_loader = create_IEMOCAP6_dataset("train", 8)
    val_loader = create_IEMOCAP6_dataset("test", 1)
    num_classes = 6
    

    model = AttentionMoE(2048, 768, num_classes)
    model = model.to(device)
    base_lr = 1e-5
    optimizer = Adam([{'params':model.parameters(), 'lr':base_lr}])

    final_val_loss = 0
    beta = 2
    kl_loss = 0.1
    for e in range(100):
        # print('Epoch-{0} lr: {1}'.format(e, optimizer.param_groups[0]['lr']))
        tot_loss, tot_acc = 0.0, 0.0
        model.train()
        pred_tr = []
        gt_tr = []
        for ind, data in enumerate(train_loader):
            model.zero_grad()
            text_feats, audio_feats, labels, _ = data
            out, aud, text, aud_out, text_out, fusion_out, div_loss, _, _ = model(audio_feats.to(device), text_feats.to(device))
            loss = compute_loss(out.to(device), labels.to(device), num_classes)
            supcon_loss = compute_supcon_loss(aud, text, labels.to(device))
            # supcon_loss_aud = compute_supcon_loss_uni(aud_feat, labels.to(device))
            # supcon_loss_text = compute_supcon_loss_uni(text_feat, labels.to(device))
            # supcon_loss = supcon_loss + supcon_loss_aud + supcon_loss_text
            loss_aud = compute_loss(aud_out.to(device), labels.to(device), num_classes)
            loss_text = compute_loss(text_out.to(device), labels.to(device), num_classes)
            loss_fusion = compute_loss(fusion_out.to(device), labels.to(device), num_classes)
            loss = loss + loss_aud + loss_text + loss_fusion + beta*supcon_loss
            # loss = beta * loss + (1-beta)*supcon_loss
            loss += kl_loss*div_loss
            # loss += 3*entropy
            # loss += compute_supcon_loss(feats, labels.to(device))
            pred, gt = compute_accuracy(out, labels)
            pred_tr.extend(pred)
            gt_tr.extend(gt)
            tot_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = 0.0, 0.0
            pred_val = []
            gt_val = []
            for ind, data in enumerate(val_loader):

                text_feats, audio_feats, labels, _ = data
                out, aud, text, aud_out, text_out, fusion_out, div_loss, _, _ = model(audio_feats.to(device), text_feats.to(device))
                loss = compute_loss(out.to(device), labels.to(device), num_classes)
                pred, gt = compute_accuracy(out.cpu(), labels)
                pred_val.extend(pred)
                gt_val.extend(gt)
                val_loss += loss.item()

            gt_tr_new = []
            pred_tr_new = []
            gt_val_new = []
            pred_val_new = []
            for i in range(len(gt_tr)):
                if gt_tr[i] < 0:
                    continue
                else:
                    gt_tr_new.append(gt_tr[i])
                    pred_tr_new.append(pred_tr[i])
            for i in range(len(gt_val)):
                if gt_val[i] < 0:
                    continue
                else:
                    gt_val_new.append(gt_val[i])
                    pred_val_new.append(pred_val[i])
            train_f1 = f1_score(gt_tr_new, pred_tr_new, average='weighted')
            val_f1 = f1_score(gt_val_new, pred_val_new, average='weighted')
            if val_f1 > final_val_loss:
                torch.save(model.state_dict(), "moe_model.pth")
                final_val_loss = val_f1
            e_log = e + 1
            
            train_loss = tot_loss/len(train_loader)
            val_loss_log = val_loss/len(val_loader)
            # print(final_val_loss)
            logger.info(f"Epoch {e_log}, \
                        Training Loss {train_loss},\
                        Training Accuracy {train_f1}")
            logger.info(f"Epoch {e_log}, \
                        Validation Loss {val_loss_log},\
                        Validation Accuracy {val_f1}")
            
def majority(aud_out, text_out, fusion_out, labels):
    # Step 1: Get class predictions
    aud_pred = aud_out.argmax(dim=-1)       # [B, T]
    text_pred = text_out.argmax(dim=-1)     # [B, T]
    fusion_pred = fusion_out.argmax(dim=-1) # [B, T]

    # Step 2: Stack predictions: [B, T, 3]
    all_preds = torch.stack([aud_pred, text_pred, fusion_pred], dim=-1)

    # Step 3: Compute majority vote
    # Count how many times each class appears in each row (dim=-1)
    # This creates a count for each possible class (up to C) at each [B, T] position
    majority_pred = []
    for i in range(all_preds.shape[0]):  # over batch
        seq_pred = []
        for j in range(all_preds.shape[1]):  # over time steps
            preds = all_preds[i, j]
            vals, counts = preds.unique(return_counts=True)
            if counts.max() > 1:
                # Majority exists
                majority_class = vals[counts.argmax()]
            else:
                # All disagree: choose fusion prediction
                majority_class = fusion_pred[i, j]
            seq_pred.append(majority_class)
        majority_pred.append(torch.stack(seq_pred))

    # Convert to tensor: [B, T]
    pred = torch.stack(majority_pred)

    pred = pred.detach().cpu().numpy().reshape(-1)
    pred = list(pred)
    labels = labels.detach().cpu().numpy().reshape(-1)
    labels = list(labels)
    return pred, labels

def get_weights(conv_name, weights, pred, gt):
    
    conv_name = conv_name[0]
    
    utt_list = conv2utt[conv_name]
    
    assert len(utt_list) == weights.size(1), f"Mismatch between utterance count and weights size."

    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        for i in range(weights.size(1)):
            utt_name = utt_list[i]
            
            audio, text, fusion = weights[0][i]
            audio = audio.item()
            text = text.item()
            fusion = fusion.item()
            
            pred_i = pred[i].item()
            gt_i = gt[i].item()
            correct = int(pred_i == gt_i)
            
            writer.writerow([conv_name, utt_name,round(audio, 4), round(text, 4), round(fusion, 4),
                correct, pred_i, gt_i])
        

def test():
    
    return_weights = False

    test_loader = create_IEMOCAP6_dataset("test", 1)
    num_classes = 6
    
    model = AttentionMoE(2048, 768, num_classes, return_weights= return_weights)
    checkpoint = torch.load('moe_model.pth', map_location = device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    # print(model.layers_comb.weight)
    model.eval()
    pred_test, gt_test = [], []
    class_names = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
    blue_green_cmap = LinearSegmentedColormap.from_list("blue_green", ["blue", "green"])
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            text_feats, audio_feats, labels, conv_names = data
            
            if return_weights:
                out, aud, text, aud_out, text_out, fusion_out, div_loss, _, _, weights = model(audio_feats.to(device), text_feats.to(device))
            else:
                out, aud, text, aud_out, text_out, fusion_out, div_loss, _, _ = model(audio_feats.to(device), text_feats.to(device))
            # out = fusion_out + aud_out + text_out
            pred, gt = compute_accuracy(out.cpu(), labels)
            # pred, gt = majority(aud_out, text_out, fusion_out, labels)
            pred_test.extend(pred)
            gt_test.extend(gt)
            
            if return_weights : 
                get_weights(conv_names, weights, pred, gt)
    gt_test_new = []
    pred_test_new = []
    for i in range(len(gt_test)):
        if gt_test[i] < 0:
            continue
        else:
            gt_test_new.append(gt_test[i])
            pred_test_new.append(pred_test[i])
    test_f1 = f1_score(gt_test_new, pred_test_new, average='weighted')
    # print(classification_report(gt_test_new, pred_test_new, output_dict=True))
    cm = confusion_matrix(gt_test_new, pred_test_new)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 24}) 
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title('6-Class Confusion Matrix')
    plt.tight_layout()
    plt.savefig("heatmap.pdf", dpi=300)
    logger.info(f"Test Accuracy {test_f1}")

def test_mod_drop():

    test_loader = create_IEMOCAP6_dataset("test", 1)
    num_classes = 6
    
    model = AttentionMoE(2048, 768, num_classes)
    checkpoint = torch.load('moe_model.pth', map_location = device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    # print(model.layers_comb.weight)
    model.eval()
    
    list_p = [0.1, 0.2, 0.3, 0.4,0.5, 0.6,0.7,0.8,0.9]
    for p in list_p:
        pred_test, gt_test = [], []
        with torch.no_grad():
            for i, data in enumerate(test_loader):

                text_feats, audio_feats, labels, _ = data
                mask = (torch.rand(text_feats.shape[1]) > p).float().unsqueeze(1)
                audio_feats = audio_feats * mask[None, :, :]
                mask = (torch.rand(text_feats.shape[1]) > p).float().unsqueeze(1)
                text_feats = text_feats * mask[None, :, :]
                out, aud, text, aud_out, text_out, fusion_out, div_loss, _, _ = model(audio_feats.to(device), text_feats.to(device))
                # out = fusion_out + aud_out + text_out
                pred, gt = compute_accuracy(out.cpu(), labels)
                # pred, gt = majority(aud_out, text_out, fusion_out, labels)
                pred_test.extend(pred)
                gt_test.extend(gt)
        gt_test_new = []
        pred_test_new = []
        for i in range(len(gt_test)):
            if gt_test[i] < 0:
                continue
            else:
                gt_test_new.append(gt_test[i])
                pred_test_new.append(pred_test[i])
        test_f1 = f1_score(gt_test_new, pred_test_new, average='weighted')
        logger.info(f"Test Accuracy {test_f1}")

def get_feats():
    
    if args.dataset == "MELD":
        train_loader = create_MELD_dataset("train", 1)
        val_loader = create_MELD_dataset("val", 1)
        test_loader = create_MELD_dataset("test", 1)
        num_classes = 7
        out_file = "text_gru_meld.pkl"
        # logits = pickle.load(open("/data2/soumyad/emo_pretraining/audio_logits_meld.pkl", "rb"))
        # logits_file = "audio_gru_logits_meld.pkl"
        train_seq = json.load(open("/home/soumyadutta/HCAM/MELD/labels/train_audio_sequence.json", "r"))
        val_seq = json.load(open("/home/soumyadutta/HCAM/MELD/labels/val_audio_sequence.json", "r"))
        test_seq = json.load(open("/home/soumyadutta/HCAM/MELD/labels/test_audio_sequence.json", "r"))
    elif args.dataset == "MOSI":
        train_loader = create_MOSI_dataset("train", 1)
        val_loader = create_MOSI_dataset("val", 1)
        test_loader = create_MOSI_dataset("test", 1)
        num_classes = 2
        out_file = "audio_gru_mosi.pkl"
        train_seq = json.load(open("/home/soumyad/TAFFC/MOSI_data/train_audio_sequence.json", "r"))
        val_seq = json.load(open("/home/soumyad/TAFFC/MOSI_data/val_audio_sequence.json", "r"))
        test_seq = json.load(open("/home/soumyad/TAFFC/MOSI_data/test_audio_sequence.json", "r"))
    elif args.dataset == "IEMOCAP4":
        train_loader = create_IEMOCAP4_dataset("train", 1)
        val_loader = create_IEMOCAP4_dataset("val", 1)
        test_loader = create_IEMOCAP4_dataset("test", 1)
        out_file = "audio_gru_iemocap4.pkl"
        logits = pickle.load(open("/data2/soumyad/emo_pretraining/audio_logits_iemocap4.pkl", "rb"))
        logits_file = "audio_gru_logits_iemocap4.pkl"
        train_seq = json.load(open("/home/soumyad/TAFFC/IEMOCAP_data/train_audio_sequence.json", "r"))
        val_seq = json.load(open("/home/soumyad/TAFFC/IEMOCAP_data/val_audio_sequence.json", "r"))
        test_seq = json.load(open("/home/soumyad/TAFFC/IEMOCAP_data/test_audio_sequence.json", "r"))
        num_classes = 4
    elif args.dataset == "IEMOCAP6":
        train_loader = create_IEMOCAP6_dataset("train", 1)
        val_loader = create_IEMOCAP6_dataset("val", 1)
        test_loader = create_IEMOCAP6_dataset("test", 1)
        out_file = "text_gru_iemocap6.pkl"
        num_classes = 6
        train_seq = json.load(open("/home/soumyadutta/HCAM/IEMOCAP6/train_audio_sequence.json", "r"))
        val_seq = json.load(open("/home/soumyadutta/HCAM/IEMOCAP6/val_audio_sequence.json", "r"))
        test_seq = json.load(open("/home/soumyadutta/HCAM/IEMOCAP6/test_audio_sequence.json", "r"))
    else:
        print("Dataset not found")

    model = GRUModel(2048, 512, num_classes, 0.2, 3, True)
    checkpoint = torch.load('text_gru_model.tar', map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    train_dict, val_dict, test_dict = dict(), dict(), dict()
    test_logits = dict()
    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader)):
            feats, _, name = data
            with torch.no_grad():
                out_feats, _ = model(feats.to(device))
            utterances = train_seq[name[0]]
            for utt_ind, utt in enumerate(utterances):
                train_dict[utt] = out_feats.squeeze(0)[utt_ind].cpu().detach().numpy()
        with open(out_file.replace(".pkl", "_train.pkl"), 'wb') as handle:
            pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


        for i, data in enumerate(tqdm(val_loader)):
            feats, _, name = data
            with torch.no_grad():
                out_feats, _ = model(feats.to(device))
            utterances = val_seq[name[0]]
            for utt_ind, utt in enumerate(utterances):
                val_dict[utt] = out_feats.squeeze(0)[utt_ind].cpu().detach().numpy()
        with open(out_file.replace(".pkl", "_valid.pkl"), 'wb') as handle:
            pickle.dump(val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for i, data in enumerate(tqdm(test_loader)):
            feats, _, name = data
            with torch.no_grad():
                out_feats, out = model(feats.to(device))
            utterances = test_seq[name[0]]
            for utt_ind, utt in enumerate(utterances):
                test_dict[utt] = out_feats.squeeze(0)[utt_ind].cpu().detach().numpy()
                # out[0, utt_ind, :] = 1*out[0, utt_ind, :] + 0*torch.tensor(logits[utt]).to(device)
                # test_logits[utt] = out.squeeze(0)[utt_ind].cpu().detach().numpy()
        with open(out_file.replace(".pkl", "_test.pkl"), 'wb') as handle:
            pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(logits_file, 'wb') as handle:
        #     pickle.dump(test_logits, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    
    # conv2utt = json.load(open("/home/soumyadutta/HCAM/IEMOCAP6/test_audio_sequence.json"))
    # CSV_FILE = "utterance_weights_IEMOCAP6.csv"
    # CSV_HEADER = ['Conversation', 'Utterance', 'Audio', 'Text', 'Fusion', 'Correct', 'Prediction', 'GT']
    # if not os.path.exists(CSV_FILE):
    #     with open(CSV_FILE, mode='w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(CSV_HEADER)
            
    train()
    test()
    # test_mod_drop()
    # get_feats()
