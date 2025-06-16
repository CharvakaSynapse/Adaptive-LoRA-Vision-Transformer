import torch
from transformers import DeiTForImageClassification
from peft import LoraConfig, get_peft_model
from collections import defaultdict
from tqdm import tqdm

def initialize_model(device, config):
    model = DeiTForImageClassification.from_pretrained(config['model']['pretrained_model'])
    model.classifier = torch.nn.Linear(model.classifier.in_features, config['model']['num_classes'])
    model.classifier.requires_grad_(True)
    model.to(device)
    return model

def compute_adaptive_ranks(model, train_loader, device, config):
    fisher_sums = defaultdict(float)
    grad_sums = defaultdict(float)
    mean_out = defaultdict(float)
    sq_mean_out = defaultdict(float)
    n_batches = 0
    hook_handles = []
    target_layers = []

    def hook_fn(name):
        def hook(module, input, output):
            output_tensor = output[0] if isinstance(output, tuple) else output
            batch_out_flat = output_tensor.detach().view(-1, output_tensor.shape[-1])
            mean_out[name] += batch_out_flat.mean(dim=0)
            sq_mean_out[name] += (batch_out_flat ** 2).mean(dim=0)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(t in name for t in ["query", "key", "value"]):
            target_layers.append(name)
            handle = module.register_forward_hook(hook_fn(name))
            hook_handles.append(handle)

    temp_optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['adamw_lr'])
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader, desc="Compute Importance", leave=False), 1):
        if i > 100: break
        images, labels = images.to(device), labels.to(device)
        temp_optimizer.zero_grad()
        loss = criterion(model(images).logits, labels)
        loss.backward()
        n_batches += 1
        for name, module in model.named_modules():
            if name in target_layers:
                for p_name, p in [("weight", module.weight), ("bias", getattr(module, 'bias', None))]:
                    if p is not None and p.grad is not None:
                        fisher_sums[f"{name}.{p_name}"] += (p.grad ** 2).mean().item()
                        grad_sums[f"{name}.{p_name}"] += p.grad.abs().mean().item()
    for handle in hook_handles: handle.remove()

    cov_trace = {name: ((sq_mean_out[name] / n_batches) - ((mean_out[name] / n_batches) ** 2)).sum().item() for name in mean_out}
    combined_importance = {}
    fisher_min, fisher_max = min(fisher_sums.values()), max(fisher_sums.values())
    grad_min, grad_max = min(grad_sums.values()), max(grad_sums.values())
    cov_min, cov_max = (min(cov_trace.values()), max(cov_trace.values())) if cov_trace else (0, 0)
    for name in target_layers:
        fisher_score = sum(fisher_sums.get(p, 0) for p in [f"{name}.weight", f"{name}.bias"])
        grad_score = sum(grad_sums.get(p, 0) for p in [f"{name}.weight", f"{name}.bias"])
        cov_score = cov_trace.get(name, 0)
        fisher_z = (fisher_score - fisher_min) / (fisher_max - fisher_min + 1e-6)
        grad_z = (grad_score - grad_min) / (grad_max - grad_min + 1e-6)
        cov_z = (cov_score - cov_min) / (cov_max - cov_min + 1e-6) if cov_trace else 0
        score = (config['model']['lora']['importance_weights']['fisher'] * fisher_z +
                 config['model']['lora']['importance_weights']['grad'] * grad_z +
                 config['model']['lora']['importance_weights']['cov'] * cov_z)
        combined_importance[name] = score

    r_max, r_min, r_total = config['model']['lora']['r_max'], config['model']['lora']['r_min'], config['model']['lora']['r_total']
    adaptive_ranks = {}
    sorted_layers = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
    remaining_budget = r_total
    for name, score in sorted_layers:
        if remaining_budget <= 0:
            adaptive_ranks[name] = r_min
        else:
            rank = min(max(r_min, int(score * r_total)), r_max, remaining_budget)
            adaptive_ranks[name] = rank
            remaining_budget -= rank
    if remaining_budget > 0:
        for name in [n for n, s in sorted_layers][:remaining_budget]:
            adaptive_ranks[name] = min(adaptive_ranks.get(name, 0) + 1, r_max)
    
    print("\nAdaptive Ranks (Attention Only):", adaptive_ranks)
    return adaptive_ranks

def apply_lora(model, adaptive_ranks, device, config):
    lora_config = LoraConfig(
        r=config['model']['lora']['r_max'],
        lora_alpha=config['model']['lora']['lora_alpha'],
        lora_dropout=config['model']['lora']['lora_dropout'],
        target_modules=list(adaptive_ranks.keys()),
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    for name, rank in adaptive_ranks.items():
        module = model.get_submodule(name)
        if hasattr(module, 'r'):
            if isinstance(module.r, dict) and 'default' in module.r:
                module.r['default'] = rank
            else:
                module.r = rank
            in_features = module.in_features
            out_features = module.out_features
            module.lora_A['default'].weight.data = torch.randn(rank, in_features).to(device) * 0.02
            module.lora_B['default'].weight.data = torch.zeros(out_features, rank).to(device)
    
    print("\nFinal trainable model structure:")
    model.print_trainable_parameters()
    return model

def set_trainable_parameters(model):
    for name, param in model.named_parameters():
        if 'lora_' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False