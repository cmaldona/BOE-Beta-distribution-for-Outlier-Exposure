import os
import time
import random
import pandas as pd
import numpy as np
from numpy.linalg import norm, pinv
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    brier_score_loss
)
from sklearn.covariance import EmpiricalCovariance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta, kl_divergence, Uniform

import transformers
from transformers import (
    RobertaModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    BertModel,
    BertTokenizer,
    AdamW
)
from scipy.special import logsumexp

from tqdm import tqdm
from peft import (
    PeftModel,
    PeftConfig,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_id = 0

def set_seed(seed):
    """Establece la semilla para reproducibilidad en diferentes módulos."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed)


def prepare_environment(seed, device):
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    set_seed(seed)
    return torch.device(device if torch.cuda.is_available() else "cpu")


def save_results(model, val_metrics, train_metrics, params, base_path, k_classes):
    params_str = ", ".join(f"{k}:{v}" for k, v in params.items() if k != 'setup')
    params_str += ", " + f"dataset: {params['setup'][0]}"
    params_str += ", " + f"k: {k_classes}"
    model_save_path = f'{base_path}/models/trained_proposal_roberta_{params["structure"]}({params_str}).pth'
    torch.save(model.state_dict(), model_save_path)

    csv_path = f'{base_path}/results/train_eval_proposal_roberta_{params["structure"]}({params_str}).csv'

    combined_df_val = pd.concat(val_metrics, ignore_index=True)
    combined_df_train = pd.concat(train_metrics, ignore_index=True)

    if not os.path.exists(csv_path):
        combined_df_val.to_csv(csv_path, index=False)
        combined_df_train.to_csv(csv_path, mode='a', index=False)
    else:
        combined_df_val.to_csv(csv_path, mode='a', header=False, index=False)
        combined_df_train.to_csv(csv_path, mode='a', header=False, index=False)

    return model_save_path


def initialize_model(structure, k_classes, peft_config, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=k_classes, return_dict=True
    )

    if structure == 'frozen':
        for param in model.base_model.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    elif structure == 'peft':
        model = get_peft_model(model, peft_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.to(device)
    return model, optimizer


class CustomLoss_draft(nn.Module):
    def __init__(self, loss_type='kl', weight_regularizer=0, num_classes=150, distribution='Beta',
                 beta_params=((0.1,10),(100,100))):
        super(CustomLoss, self).__init__()
        self.loss_type = loss_type
        self.weight_regularizer = weight_regularizer
        self.cross_entropy = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.distribution = distribution
        self.alpha1, self.beta1 = beta_params[0]
        self.alpha2, self.beta2 = beta_params[1]

    def combined_beta_sample(self, sample_size):
        samples_beta1 = Beta(self.alpha1, self.beta1).sample((sample_size, self.num_classes))
        samples_beta2 = Beta(self.alpha2, self.beta2).sample((sample_size, self.num_classes))
        combined_samples = samples_beta1 + samples_beta2
        combined_samples /= combined_samples.sum(dim=1, keepdim=True)
        return combined_samples

    def calculate_loss(self, dist_a, dist_b):
        if self.loss_type == 'kl':
            return kl_divergence(dist_a, dist_b).mean()
        elif self.loss_type == 'js':
            m = 0.5 * (dist_a.probs + dist_b.probs)
            return 0.5 * (kl_divergence(dist_a, Categorical(probs=m)) + kl_divergence(dist_b,
                                                                                      Categorical(probs=m))).mean()
        elif self.loss_type == 'tv':
            return 0.5 * torch.abs(dist_a.probs - dist_b.probs).sum(dim=1).mean()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def forward(self, model_output, target, model_ood_output):
        softmax_output = torch.softmax(model_output, dim=1)
        softmax_special_output = torch.softmax(model_ood_output, dim=1).to(device)
        dist_a = Categorical(probs=softmax_special_output)

        if self.distribution == 'Beta':
            combined_beta_samples = self.combined_beta_sample(softmax_special_output.shape[0]).to(device)
            dist_b = Categorical(probs=combined_beta_samples)
        elif self.distribution == 'Uniform':
            uniform_distribution = Uniform(torch.zeros_like(softmax_special_output),
                                           torch.ones_like(softmax_special_output))
            uniform_samples = uniform_distribution.sample()
            dist_b = Categorical(probs=uniform_samples)

        specific_loss = self.calculate_loss(dist_a, dist_b)
        ce_loss = self.cross_entropy(model_output, target)
        total_loss = ce_loss + (self.weight_regularizer * specific_loss)

        return total_loss


class CustomLoss_draft2(nn.Module):
    def __init__(self, loss_type='kl', weight_regularizer=0, num_classes=150, distribution='Beta',
                 beta_params=((0.1,10),(100,100))):
        super(CustomLoss, self).__init__()
        self.loss_type = loss_type
        self.weight_regularizer = weight_regularizer
        self.cross_entropy = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.distribution = distribution
        self.alpha1, self.beta1 = beta_params[0]
        self.alpha2, self.beta2 = beta_params[1]

        # Generar distribuciones y guardarlas en variables
        if self.distribution == 'Beta':
            self.combined_beta_samples = self._create_combined_beta_sample()
        elif self.distribution == 'Uniform':
            self.uniform_samples = self._create_uniform_sample()

    def _create_combined_beta_sample(self):
        """Crea y almacena muestras combinadas de distribuciones Beta."""
        samples_beta1 = Beta(self.alpha1, self.beta1).sample((1, self.num_classes))
        samples_beta2 = Beta(self.alpha2, self.beta2).sample((1, self.num_classes))
        combined_samples = samples_beta1 + samples_beta2
        combined_samples /= combined_samples.sum(dim=1, keepdim=True)
        return combined_samples.to(device)

    def _create_uniform_sample(self):
        """Crea y almacena muestras de distribución uniforme."""
        uniform_distribution = Uniform(torch.zeros(1, self.num_classes), torch.ones(1, self.num_classes))
        return uniform_distribution.sample().to(device)

    def calculate_loss(self, dist_a, dist_b):
        if self.loss_type == 'kl':
            return kl_divergence(dist_a, dist_b).mean()
        elif self.loss_type == 'js':
            m = 0.5 * (dist_a.probs + dist_b.probs)
            return 0.5 * (kl_divergence(dist_a, Categorical(probs=m)) + kl_divergence(dist_b, Categorical(probs=m))).mean()
        elif self.loss_type == 'tv':
            return 0.5 * torch.abs(dist_a.probs - dist_b.probs).sum(dim=1).mean()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def forward(self, model_output, target, model_ood_output):
        softmax_output = torch.softmax(model_output, dim=1)
        softmax_special_output = torch.softmax(model_ood_output, dim=1).to(device)
        dist_a = Categorical(probs=softmax_special_output)

        if self.distribution == 'Beta':
            dist_b = Categorical(probs=self.combined_beta_samples.repeat(softmax_special_output.shape[0], 1))
        elif self.distribution == 'Uniform':
            dist_b = Categorical(probs=self.uniform_samples.repeat(softmax_special_output.shape[0], 1))

        specific_loss = self.calculate_loss(dist_a, dist_b)
        ce_loss = self.cross_entropy(model_output, target)
        total_loss = ce_loss + (self.weight_regularizer * specific_loss)

        return total_loss


class CustomLoss(nn.Module):
    def __init__(self, loss_type='kl', weight_regularizer=0, num_classes=150, distribution='Beta',
                 beta_params=((0.1, 10), (100, 100))):
        super(CustomLoss, self).__init__()
        self.loss_type = loss_type
        self.weight_regularizer = weight_regularizer
        self.cross_entropy = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.distribution = distribution
        self.alpha1, self.beta1 = beta_params[0]
        self.alpha2, self.beta2 = beta_params[1]

        # Crear muestras uniformes una sola vez
        if self.distribution == 'Uniform':
            self.uniform_samples = self._create_uniform_sample()

    def _create_combined_beta_sample(self):
        """Crea y devuelve muestras combinadas de distribuciones Beta sin seguir los gradientes."""
        with torch.no_grad():  # Desactiva el seguimiento de gradientes
            samples_beta1 = Beta(self.alpha1, self.beta1).sample((1, self.num_classes))
            samples_beta2 = Beta(self.alpha2, self.beta2).sample((1, self.num_classes))
            combined_samples = samples_beta1 + samples_beta2
            combined_samples /= combined_samples.sum(dim=1, keepdim=True)
        return combined_samples.to(device)

    def _create_uniform_sample(self):
        """Crea y devuelve muestras de distribución uniforme."""
        uniform_distribution = Uniform(torch.zeros(1, self.num_classes), torch.ones(1, self.num_classes))
        return uniform_distribution.sample().to(device)

    def calculate_loss(self, dist_a, dist_b):
        if self.loss_type == 'kl':
            return kl_divergence(dist_a, dist_b).mean()
        elif self.loss_type == 'js':
            m = 0.5 * (dist_a.probs + dist_b.probs)
            return 0.5 * (kl_divergence(dist_a, Categorical(probs=m)) + kl_divergence(dist_b, Categorical(probs=m))).mean()
        elif self.loss_type == 'tv':
            return 0.5 * torch.abs(dist_a.probs - dist_b.probs).sum(dim=1).mean()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def forward(self, model_output, target, model_ood_output):
        softmax_output = torch.softmax(model_output, dim=1)
        softmax_special_output = torch.softmax(model_ood_output, dim=1).to(device)
        dist_a = Categorical(probs=softmax_special_output)

        if self.distribution == 'Beta':
            combined_beta_samples = self._create_combined_beta_sample()
            dist_b = Categorical(probs=combined_beta_samples.repeat(softmax_special_output.shape[0], 1))
        elif self.distribution == 'Uniform':
            dist_b = Categorical(probs=self.uniform_samples.repeat(softmax_special_output.shape[0], 1))

        specific_loss = self.calculate_loss(dist_a, dist_b)
        ce_loss = self.cross_entropy(model_output, target)
        total_loss = ce_loss + (self.weight_regularizer * specific_loss)

        return total_loss

        

def calculate_metrics(outputs, y, pred, gen_mask):
    gx = np.array(torch.stack(outputs))
    true_labels = np.array(torch.stack(y))
    pred_labels = np.array(torch.stack(pred))
    mask_gen = np.array(torch.stack(gen_mask))

    exps = [
        (1, -2),
        (1, -1),
        (-1, -2),
        (1, (-1, -2))
    ]

    metrics = {}
    mask_id = np.isin(mask_gen, 1)
    metrics['acc'] = round(accuracy_score(true_labels[mask_id], pred_labels[mask_id]), 3)
    metrics['f1_m'] = round(f1_score(true_labels[mask_id], pred_labels[mask_id], average='micro'), 3)
    metrics['f1_M'] = round(f1_score(true_labels[mask_id], pred_labels[mask_id], average='macro'), 3)

    inverse_code = {
        -2: 'far',
        -1: 'near',
        1: 'ID'
    }

    for j, (pos, neg) in enumerate(exps):
        gen_is_pos = np.isin(mask_gen, pos)
        gen_is_neg = np.isin(mask_gen, neg)

        bin_labels = np.concatenate([np.ones(np.sum(gen_is_pos)), np.zeros(np.sum(gen_is_neg))])
        pos_scores, neg_scores = np.array(gx)[gen_is_pos], np.array(gx)[gen_is_neg]
        bin_scores = np.concatenate([pos_scores, neg_scores])
        if len(bin_scores.shape) > 1:
            if bin_scores.shape[1] > 1:
                bin_scores = np.max(bin_scores, axis=1)

        auroc = roc_auc_score(bin_labels, bin_scores)
        aupr = average_precision_score(bin_labels, bin_scores)
        fpr, tpr, thresholds = roc_curve(bin_labels, bin_scores)
        fpr_95 = fpr[np.argmax(tpr >= 0.95)]

        pos_str, neg_str = map(
            lambda x: '+'.join(map(inverse_code.get, x)) if isinstance(x, tuple) else inverse_code[x], [pos, neg])
        metrics.update({
            f'AUROC_{pos_str}/{neg_str}': round(auroc, 3),
            f'FPR95_{pos_str}/{neg_str}': round(fpr_95, 3)
        })

    return metrics


def train_model(model, dataloader, optimizer, custom_loss_function):
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_probs, all_gen = [], [], [], []

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        indices_id = (batch['generalisation'] == 1).nonzero().squeeze()
        indices_near_ood = (batch['generalisation'] == -1).nonzero().squeeze()
        indices_far_ood = (batch['generalisation'] == -2).nonzero().squeeze()

        ids_id = batch['input_ids'][indices_id].to(device)
        masks_id = batch['attention_mask'][indices_id].to(device)
        labels_id = batch['labels'][indices_id].to(device)
        ids_id = ids_id.unsqueeze(0) if ids_id.dim() == 1 else ids_id
        masks_id = masks_id.unsqueeze(0) if masks_id.dim() == 1 else masks_id

        ids_near = batch['input_ids'][indices_near_ood].to(device)
        masks_near = batch['attention_mask'][indices_near_ood].to(device)
        ids_near = ids_near.unsqueeze(0) if ids_near.dim() == 1 else ids_near
        masks_near = masks_near.unsqueeze(0) if masks_near.dim() == 1 else masks_near

        ids_far = batch['input_ids'][indices_far_ood].to(device)
        masks_far = batch['attention_mask'][indices_far_ood].to(device)
        ids_far = ids_far.unsqueeze(0) if ids_far.dim() == 1 else ids_far
        masks_far = masks_far.unsqueeze(0) if masks_far.dim() == 1 else masks_far

        outputs = model(input_ids=ids_id, attention_mask=masks_id)
        near_outputs_ood = model(input_ids=ids_near, attention_mask=masks_near)
        far_outputs_ood = model(input_ids=ids_far, attention_mask=masks_far)

        logits_near_ood = near_outputs_ood.logits
        logits_far_ood = far_outputs_ood.logits
        outputs_ood_logits = torch.cat([logits_near_ood, logits_far_ood], dim=0)

        loss = custom_loss_function(outputs.logits, labels_id, outputs_ood_logits)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        outputs = torch.cat([outputs.logits, outputs_ood_logits])
        probs = F.softmax(outputs, dim=-1).detach().cpu()
        preds = outputs.argmax(dim=-1).detach().cpu()

        labels = batch["labels"].detach().cpu()
        generalizations = batch["generalisation"].detach().cpu()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_gen.extend(generalizations)

        torch.cuda.empty_cache()

    metrics = calculate_metrics(
        all_probs,
        all_labels,
        all_preds,
        all_gen
    )

    avg_loss = total_loss / len(dataloader)
    metrics['avg_loss'] = round(avg_loss, 1)

    torch.cuda.empty_cache()

    return metrics


def eval_model(model, eval_dataloader):
    model.eval()
    outputs = []
    pred = []
    gen_mask = []
    y = []

    with torch.no_grad():
        for b in eval_dataloader:
            ids = b['input_ids'].to(device)
            masks = b['attention_mask'].to(device)
            gens = b['generalisation'].detach().cpu()
            y_true = b['labels'].detach().cpu()

            out = model(input_ids=ids, attention_mask=masks)
            outputs.extend(out.logits.detach().cpu())
            pred.extend(out.logits.argmax(dim=-1).detach().cpu())
            gen_mask.extend(gens)
            y.extend(y_true)
            torch.cuda.empty_cache()

    metrics = calculate_metrics(
        outputs,
        y,
        pred,
        gen_mask
    )

    return metrics


def train_evaluate(model, train_dataloader, eval_dataloader, optimizer, num_epochs, flag_peft, custom_loss_function):
    epoch_times = []
    best_val_acc = -1
    epochs_without_improvement = 0
    max_epochs_without_improvement = 5
    val_metrics_epoch = []
    train_metrics_epoch = []

    for epoch in range(num_epochs):

        start_time = time.time()
        train_metrics = train_model(model, train_dataloader, optimizer, custom_loss_function)
        end_time = time.time()
        epoch_time = end_time - start_time
        train_metrics['time_epoch'] = epoch_time
        epoch_times.append(epoch_time)

        start_time = time.time()
        val_metrics = eval_model(model, eval_dataloader)
        end_time = time.time()
        epoch_time = end_time - start_time
        val_metrics['time_epoch'] = epoch_time

        val_metrics_epoch.append(pd.DataFrame([val_metrics]))
        train_metrics_epoch.append(pd.DataFrame([train_metrics]))

        print(f"Epoch {epoch}: \nTrain metrics {train_metrics} \n Val metrics {val_metrics}\n")

        current_val_acc = val_metrics['acc']

        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'best_model_state.pth')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > max_epochs_without_improvement:
            print(f"Early stopping at Epoch: {epoch}\n", )
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return val_metrics_epoch, train_metrics_epoch


features_handle = []


def get_activation(name):
    def hook(model, input, output):
        features_handle.append(output.detach())

    return hook


def get_vim_model(model, train_dataloader, dim=10):
    global features_handle
    features_handle.clear()

    hook = model.base_model.model.classifier.modules_to_save['default'].dense.register_forward_hook(
        get_activation('dense'))

    labs = []
    features = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            indices_id = (batch['generalisation'] == 1).nonzero().squeeze()

            ids_id = batch['input_ids'][indices_id].to(device)
            masks_id = batch['attention_mask'][indices_id].to(device)
            ids_id = ids_id.unsqueeze(0) if ids_id.dim() == 1 else ids_id
            masks_id = masks_id.unsqueeze(0) if masks_id.dim() == 1 else masks_id

            dummy = model(input_ids=ids_id, attention_mask=masks_id)
            penultimate_layer = features_handle[-1].detach().cpu()

            features.extend(penultimate_layer.tolist())
            labs.extend(batch["labels"][indices_id])

            # Liberar memoria
            del ids_id, masks_id, dummy, penultimate_layer
            torch.cuda.empty_cache()
            features_handle.clear()

    features = np.array(features)
    labs = np.array(labs)

    w = model.classifier.modules_to_save['default'].out_proj.weight.data.cpu().numpy()
    b = model.classifier.modules_to_save['default'].out_proj.bias.data.cpu().numpy()
    logit_id_train = features @ w.T + b

    hook.remove()

    u = -np.matmul(pinv(w), b)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(features - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[dim:]]).T
    )
    vlogit_id_train = norm(
        np.matmul(features - u, NS),
        axis=-1
    )
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()

    return u, NS, alpha


def test_model_vim(model, dataloader, alpha, u, NS):
    global features_handle
    features_handle.clear()
    model.eval()

    outputs = []
    pred = []
    gen_mask = []
    y = []

    hook = model.base_model.model.classifier.modules_to_save['default'].dense.register_forward_hook(
        get_activation('dense'))
    w = model.classifier.modules_to_save['default'].out_proj.weight.data.cpu().numpy()
    b = model.classifier.modules_to_save['default'].out_proj.bias.data.cpu().numpy()

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            gens = batch['generalisation'].detach().cpu()
            y_true = batch['labels'].detach().cpu()

            out = model(input_ids=ids, attention_mask=masks)
            penultimate_layer = features_handle[-1].detach().cpu()
            logits = penultimate_layer @ w.T + b
            _, preds = torch.max(logits, dim=1)
            energy = logsumexp(logits.numpy(), axis=-1)
            vlogit_ood = alpha * norm(
                np.matmul(penultimate_layer.numpy() - u, NS),
                axis=-1
            )
            score_ood = -vlogit_ood + energy

            outputs.extend(torch.Tensor(score_ood))
            pred.extend(out.logits.argmax(dim=-1).detach().cpu())
            gen_mask.extend(gens)
            y.extend(y_true)
            
            # Liberar memoria
            del ids, masks, out, penultimate_layer, logits
            torch.cuda.empty_cache()
            features_handle.clear()

    metrics = calculate_metrics(
        outputs,
        y,
        pred,
        gen_mask
    )

    hook.remove()

    return metrics