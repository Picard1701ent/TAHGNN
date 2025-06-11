import logging
from types import SimpleNamespace

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

import config
from models.model import AHGNN
from utils import load_data

cfg = SimpleNamespace(**vars(config))
device = torch.device(cfg.device)


def evaluate_result(out, labels):
    prob = torch.exp(out)
    preds = torch.argmax(prob, dim=1)

    probs_pos_class = prob[:, 1].detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    auc = roc_auc_score(labels_np, probs_pos_class)
    accuracy = accuracy_score(labels_np, preds.cpu().numpy())
    f1 = f1_score(labels_np, preds.cpu().numpy())
    tn, fp, fn, tp = confusion_matrix(labels_np, preds.cpu().numpy()).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, f1, sensitivity, specificity, auc


def train_one_epoch(model, optimizer, dataloader, loss_func):
    total_loss = 0.0
    total = 0
    total_out = []
    total_labels = []
    total_e_fts = []

    model.train()

    for data, label, H, node_strength, gloable_efficiency, dist_matrix in dataloader:
        data = data.to(device)

        label = label.to(device)
        H = H.to(device)
        node_strength = node_strength.to(device)
        gloable_efficiency = gloable_efficiency.to(device)

        optimizer.zero_grad()
        out, e_fts = model(data, H, node_strength, gloable_efficiency, dist_matrix)
        loss = loss_func(out, label)
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
        total += label.shape[0]
        total_out.append(out)
        total_labels.append(label)
        total_e_fts.append(e_fts)
    total_out = torch.cat(total_out, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    total_e_fts = torch.cat(total_e_fts, dim=0)
    accuracy, f1, sensitivity, specificity, auc = evaluate_result(
        total_out, total_labels
    )
    return (
        accuracy,
        f1,
        sensitivity,
        specificity,
        auc,
        total_loss,
        total_e_fts,
        total_labels,
    )


def evaluate_model(model, dataloader, loss_func):
    model.eval()
    total_loss = 0.0
    total = 0
    total_prob = []
    total_label = []
    total_e_fts = []
    with torch.no_grad():
        for (
            data,
            label,
            H,
            node_strength,
            gloable_efficiency,
            dist_matrix,
        ) in dataloader:
            data = data.to(device)
            label = label.to(device)
            H = H.to(device)
            node_strength = node_strength.to(device)
            gloable_efficiency = gloable_efficiency.to(device)
            out, e_fts = model(data, H, node_strength, gloable_efficiency, dist_matrix)
            loss = loss_func(out, label)
            total_loss = loss.item() + total_loss
            total = total + data.shape[0]
            total_prob.append(out)
            total_label.append(label)
            total_e_fts.append(e_fts)
    total_prob = torch.cat(total_prob, dim=0)
    total_label = torch.cat(total_label, dim=0)
    total_e_fts = torch.cat(total_e_fts, dim=0)
    accuracy, f1, sensitivity, specificity, auc = evaluate_result(
        total_prob, total_label
    )
    return (
        accuracy,
        f1,
        sensitivity,
        specificity,
        auc,
        total_loss,
        total_e_fts,
        total_label,
    )


def train():
    datas, labels, structures, Hs, node_strengths, gloable_efficiencys, dist_matrixs = (
        load_data()
    )
    datas = torch.from_numpy(datas).to(torch.float32)
    labels = torch.from_numpy(labels).to(torch.long)
    Hs = torch.from_numpy(Hs).to(torch.float32)
    node_strengths = torch.from_numpy(node_strengths).to(torch.float32)
    gloable_efficiencys = torch.from_numpy(gloable_efficiencys).to(torch.float32)
    dist_matrixs = torch.from_numpy(dist_matrixs).to(torch.float32)
    dataset = TensorDataset(
        datas, labels, Hs, node_strengths, gloable_efficiencys, dist_matrixs
    )
    kf = StratifiedKFold(n_splits=cfg.N, shuffle=True, random_state=cfg.seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(datas, labels)):
        text = f"Fold {fold}: \n Train index:{train_idx} \n Val index:{val_idx}"
        logging.info(text)
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=True)
        model = AHGNN()
        loss_func = torch.nn.NLLLoss(reduction="mean")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epoch // 5, eta_min=1e-7)
        model = model.to(device)
        model.train()
        for i in range(cfg.epoch):
            accuracy, f1, sensitivity, specificity, auc, ave_loss, e_fts, out_labels = (
                train_one_epoch(model, optimizer, train_loader, loss_func)
            )
            scheduler.step()
            logging.info(
                f"(Train) | Epoch={i:03d}, loss={ave_loss:.4f}, "
                f"train_acc={(accuracy * 100):.2f}, train_specificity={(specificity * 100):.2f}, "
                f"train_sensitivity={(sensitivity * 100):.2f},train_auc={(auc * 100):.2f}"
            )

            (
                accuracy,
                f1,
                sensitivity,
                specificity,
                auc,
                total_loss,
                e_fts,
                out_labels,
            ) = evaluate_model(model, val_loader, loss_func)
            text = (
                f"(Evaluate) | Epoch={i:03d}, loss={total_loss:.4f}, "
                f"test_acc={(accuracy * 100):.2f}, test_specificity={(specificity * 100):.2f}, "
                f"test_sensitivity={(sensitivity * 100):.2f},test_auc={(auc * 100):.2f},f1={(f1 * 100):.2f}"
            )
            logging.info(text)
