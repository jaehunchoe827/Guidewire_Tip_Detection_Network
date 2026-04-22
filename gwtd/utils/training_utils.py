import torch
import numpy as np
import matplotlib
# Use a non-interactive backend to avoid Tkinter initialization in forked processes
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

def generate_optimizer(model, optimizer_config):
    optimizer_name = optimizer_config['name'].lower()
    optimizer_args = optimizer_config['args']
    if optimizer_name == 'adamw':
        params = set_params(model, optimizer_config['weight_decay'])
        if optimizer_args is not None:
            return torch.optim.AdamW(params, **optimizer_args)
        return torch.optim.AdamW(params)
    elif optimizer_name == 'sgd':
        params = set_params(model, optimizer_config['weight_decay'])
        if optimizer_args is not None:
            return torch.optim.SGD(params, **optimizer_args)
        return torch.optim.SGD(params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def set_params(model, decay):
    p1 = []
    p2 = []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=0):
            if not p.requires_grad:
                continue
            if n == "bias":  # bias (no decay)
                p1.append(p)
            elif n == "weight" and isinstance(m, norm):  # norm-weight (no decay)
                p1.append(p)
            else:
                p2.append(p)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


def generate_lr_scheduler(epochs, num_steps_per_epoch, lr_scheduler_config):
    lr_scheduler_name = lr_scheduler_config['name'].lower()
    lr_scheduler_args = lr_scheduler_config['args']
    if lr_scheduler_name == 'linearlr':
        return LinearLR(epochs, num_steps_per_epoch, **lr_scheduler_args)
    elif lr_scheduler_name == 'cosinelr':
        return CosineLR(epochs, num_steps_per_epoch, **lr_scheduler_args)
    elif lr_scheduler_name == 'exponentiallr':
        return ExponentialLR(epochs, num_steps_per_epoch, **lr_scheduler_args)
    elif lr_scheduler_name == 'steplr':
        return StepLR(epochs, num_steps_per_epoch, **lr_scheduler_args)
    elif lr_scheduler_name == 'constantlr':
        return ConstantLR(epochs, num_steps_per_epoch, **lr_scheduler_args)
    elif lr_scheduler_name == 'doublecosinelr':
        return DoubleCosineLR(epochs, num_steps_per_epoch, **lr_scheduler_args)
    else:
        raise ValueError(f"Unsupported learning rate scheduler: {lr_scheduler_name}")

class LinearLR:
    def __init__(self, epochs, num_steps_per_epoch, max_lr, min_lr,
                 warmup_epochs, unfreeze_backbone_epochs=None):
        warmup_steps = int(warmup_epochs * num_steps_per_epoch)
        decay_steps = int(epochs * num_steps_per_epoch - warmup_steps)
        warmup_lr = np.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = np.linspace(max_lr, min_lr, decay_steps)
        self.total_lr = np.concatenate((warmup_lr, decay_lr))
        # if we have a unfreeze_backbone_epochs that is not None,
        # then for the steps with in the range:
        # [(unfreeze_backbone_epochs-1) * num_steps_per_epoch,
        #  (unfreeze_backbone_epochs-1 + warmup_epochs) * num_steps_per_epoch],
        # the lr should be set to
        # unfreeze_lr = (step - (unfreeze_backbone_epochs-1) * num_steps_per_epoch) / warmup_steps * max_lr
        # lr = min(lr, unfreeze_lr)
        if unfreeze_backbone_epochs is not None and warmup_steps > 0:
            unfreeze_start_step = int((unfreeze_backbone_epochs - 1) * num_steps_per_epoch)
            unfreeze_end_step = unfreeze_start_step + warmup_steps
            total_steps = self.total_lr.shape[0]
            # Clamp to valid range in case unfreeze is near/after training end
            start = max(0, unfreeze_start_step)
            end = min(total_steps, unfreeze_end_step)
            if start < end:  # valid range
                for step_idx in range(start, end):
                    progress = (step_idx - unfreeze_start_step) / warmup_steps
                    progress = max(0.0, progress)
                    unfreeze_lr = progress * (max_lr - min_lr) + min_lr
                    self.total_lr[step_idx] = min(self.total_lr[step_idx], unfreeze_lr)

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]

class CosineLR:
    def __init__(self, epochs, num_steps_per_epoch, max_lr, min_lr,
                 warmup_epochs, unfreeze_backbone_epochs=None):
        warmup_steps = int(warmup_epochs * num_steps_per_epoch)
        decay_steps = int(epochs * num_steps_per_epoch - warmup_steps)
        warmup_lr = np.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = []
        for step in range(decay_steps):
            alpha = np.cos(np.pi * step / (decay_steps-1))
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))
        self.total_lr = np.concatenate((warmup_lr, decay_lr))
        # unfreeze with warmup
        if unfreeze_backbone_epochs is not None and warmup_steps > 0:
            unfreeze_start_step = int((unfreeze_backbone_epochs - 1) * num_steps_per_epoch)
            unfreeze_end_step = unfreeze_start_step + warmup_steps
            total_steps = self.total_lr.shape[0]
            # Clamp to valid range in case unfreeze is near/after training end
            start = max(0, unfreeze_start_step)
            end = min(total_steps, unfreeze_end_step)
            if start < end:  # valid range
                for step_idx in range(start, end):
                    progress = (step_idx - unfreeze_start_step) / warmup_steps
                    progress = max(0.0, progress)
                    unfreeze_lr = progress * (max_lr - min_lr) + min_lr
                    self.total_lr[step_idx] = min(self.total_lr[step_idx], unfreeze_lr)

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]

class ExponentialLR:
    def __init__(self, epochs, num_steps_per_epoch, max_lr, min_lr,
                 warmup_epochs, gamma=0.1, unfreeze_backbone_epochs=None):
        warmup_steps = int(warmup_epochs * num_steps_per_epoch)
        decay_steps = int(epochs * num_steps_per_epoch - warmup_steps)
        warmup_lr = np.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = []
        gamma_actual = np.log(max_lr / min_lr) / (decay_steps-1)
        for step in range(decay_steps):
            lr = max_lr * np.exp(-gamma_actual * step)
            decay_lr.append(lr)
        self.total_lr = np.concatenate((warmup_lr, decay_lr))
        # unfreeze with warmup
        if unfreeze_backbone_epochs is not None and warmup_steps > 0:
            unfreeze_start_step = int((unfreeze_backbone_epochs - 1) * num_steps_per_epoch)
            unfreeze_end_step = unfreeze_start_step + warmup_steps
            total_steps = self.total_lr.shape[0]
            # Clamp to valid range in case unfreeze is near/after training end
            start = max(0, unfreeze_start_step)
            end = min(total_steps, unfreeze_end_step)
            if start < end:  # valid range
                for step_idx in range(start, end):
                    progress = (step_idx - unfreeze_start_step) / warmup_steps
                    progress = max(0.0, progress)
                    unfreeze_lr = progress * (max_lr - min_lr) + min_lr
                    self.total_lr[step_idx] = min(self.total_lr[step_idx], unfreeze_lr)

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class StepLR:
    def __init__(self, epochs, num_steps_per_epoch, max_lr, min_lr,
                 warmup_epochs, n_steps = 5, unfreeze_backbone_epochs=None):
        warmup_steps = int(warmup_epochs * num_steps_per_epoch)
        warmup_lr = np.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        total_steps = epochs * num_steps_per_epoch
        one_segment_steps = int((total_steps - warmup_steps) / n_steps)
        # generate n_steps lr that is linearly interpolated
        # between min_lr, and max_lr in log scale
        log_min_lr = np.log(min_lr)
        log_max_lr = np.log(max_lr)
        # Generate n_steps linearly spaced learning rates in log scale
        steps_lrs = np.exp(np.linspace(log_max_lr, log_min_lr, n_steps))
        lr_segments = []
        for i in range(n_steps):
            num_steps_in_current_segment = one_segment_steps
            if i == n_steps - 1:
                num_steps_in_current_segment = total_steps - warmup_steps - (n_steps - 1) * one_segment_steps
            lr_segment = np.ones(num_steps_in_current_segment) * steps_lrs[i]
            lr_segments.append(lr_segment)
        self.total_lr = np.concatenate((warmup_lr, np.concatenate(lr_segments)))
        print(f"total_steps: {total_steps}, total_lr.shape: {self.total_lr.shape}")
        # unfreeze with warmup
        if unfreeze_backbone_epochs is not None and warmup_steps > 0:
            unfreeze_start_step = int((unfreeze_backbone_epochs - 1) * num_steps_per_epoch)
            unfreeze_end_step = unfreeze_start_step + warmup_steps
            total_steps = self.total_lr.shape[0]
            # Clamp to valid range in case unfreeze is near/after training end
            start = max(0, unfreeze_start_step)
            end = min(total_steps, unfreeze_end_step)
            if start < end:  # valid range
                for step_idx in range(start, end):
                    progress = (step_idx - unfreeze_start_step) / warmup_steps
                    progress = max(0.0, progress)
                    unfreeze_lr = progress * (max_lr - min_lr) + min_lr
                    self.total_lr[step_idx] = min(self.total_lr[step_idx], unfreeze_lr)

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]

class ConstantLR:
    def __init__(self, epochs, num_steps_per_epoch, max_lr, min_lr,
                 warmup_epochs, unfreeze_backbone_epochs=None):
        warmup_steps = int(warmup_epochs * num_steps_per_epoch)
        warmup_lr = np.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        self.total_lr = np.concatenate((warmup_lr, np.ones(epochs * num_steps_per_epoch - warmup_steps) * max_lr))
        # unfreeze with warmup
        if unfreeze_backbone_epochs is not None and warmup_steps > 0:
            unfreeze_start_step = int((unfreeze_backbone_epochs - 1) * num_steps_per_epoch)
            unfreeze_end_step = unfreeze_start_step + warmup_steps
            total_steps = self.total_lr.shape[0]
            # Clamp to valid range in case unfreeze is near/after training end
            start = max(0, unfreeze_start_step)
            end = min(total_steps, unfreeze_end_step)
            if start < end:  # valid range
                for step_idx in range(start, end):
                    progress = (step_idx - unfreeze_start_step) / warmup_steps
                    progress = max(0.0, progress)
                    unfreeze_lr = progress * (max_lr - min_lr) + min_lr
                    self.total_lr[step_idx] = min(self.total_lr[step_idx], unfreeze_lr)

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class DoubleCosineLR:
    def __init__(self, epochs, num_steps_per_epoch, max_lr, min_lr,
                 warmup_epochs, unfreeze_backbone_epochs=None):
        """
        DoubleCosine LR scheduler. This is equivalent to two cosine LRs in series,
        One from the start to unfreeze_backbone_epochs, and one from unfreeze_backbone_epochs to the end.
        If unfreeze_backbone_epochs is None or total epoch is less than unfreeze_backbone_epochs,
        then this is equivalent to a single cosine LR.
        """
        if unfreeze_backbone_epochs is None or unfreeze_backbone_epochs > epochs or unfreeze_backbone_epochs <= 1:
            self.total_lr = CosineLR(epochs, num_steps_per_epoch, max_lr, min_lr, warmup_epochs).total_lr
            return
        warmup_steps = int(warmup_epochs * num_steps_per_epoch)
        first_decay_steps = int((unfreeze_backbone_epochs - 1) * num_steps_per_epoch - warmup_steps)
        second_decay_steps = int(epochs * num_steps_per_epoch - first_decay_steps - 2*warmup_steps)
        first_warmup_lr = np.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        second_warmup_lr = np.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        first_decay_lr = []
        second_decay_lr = []
        for step in range(first_decay_steps):
            alpha = np.cos(np.pi * step / (first_decay_steps)) # for the first decay, we do not include the min_lr
            first_decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))
        for step in range(second_decay_steps):
            alpha = np.cos(np.pi * step / (second_decay_steps-1))
            second_decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))
        self.total_lr = np.concatenate((first_warmup_lr, first_decay_lr, second_warmup_lr, second_decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


def plot_lr_scheduler(lr_scheduler, save_dir):
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(lr_scheduler.total_lr, label='Learning Rate')
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate (log scale)')
    plt.title('Learning Rate Scheduler')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight')
    plt.close()