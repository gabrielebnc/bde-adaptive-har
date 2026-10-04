"""Three-exit residual CNN, including real per-sample conditional execution."""
from dataclasses import dataclass

import torch
from torch import nn

# These are separate fixed architectures selected before training. Width never
# changes at runtime; each checkpoint contains only one architecture's weights.
MODEL_SIZES = {
    "tiny_5k": (7, 10, 15),
    "micro": (4, 8, 16),
    "tiny_25k": (4, 7, 7),
    "tiny_1k": (4, 5, 6),
}
STAGE_BLOCKS = {"tiny_5k": (3, 3, 1), "micro": (2, 2, 2),
                "tiny_25k": (4, 3, 2), "tiny_1k": (2, 2, 1)}
STEM_CHANNELS = {"tiny_1k": 1}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1 else
                     nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                                   nn.BatchNorm1d(out_channels)))

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)) + residual)


def classifier(channels, dropout, num_classes):
    return nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                         nn.Dropout(dropout), nn.Linear(channels, num_classes))


@dataclass
class AdaptiveOutput:
    logits: torch.Tensor
    predictions: torch.Tensor
    confidence: torch.Tensor
    exit_indices: torch.Tensor  # 1, 2, or 3; also the number of stages executed.


class AdaptiveHAR(nn.Module):
    def __init__(self, num_classes=6, dropouts=(0.20, 0.25, 0.30), model_size="tiny_25k",
                 final_mode="independent"):
        super().__init__()
        if len(dropouts) != 3:
            raise ValueError("Three dropout rates are required")
        if model_size not in MODEL_SIZES:
            raise ValueError(f"Unknown model_size {model_size!r}; choose {list(MODEL_SIZES)}")
        if final_mode not in {"independent", "residual"}:
            raise ValueError("final_mode must be independent or residual")
        c1, c2, c3 = MODEL_SIZES[model_size]
        b1, b2, b3 = STAGE_BLOCKS[model_size]
        stem_channels = STEM_CHANNELS.get(model_size, c1)
        self.config = {"num_classes": num_classes, "dropouts": list(dropouts),
                       "model_size": model_size, "final_mode": final_mode}
        self.stem = nn.Sequential(nn.Conv1d(9, stem_channels, 7, 2, 3, bias=False),
                                  nn.BatchNorm1d(stem_channels), nn.ReLU(inplace=True))
        def stage(in_channels, out_channels, stride, blocks):
            return nn.Sequential(ResidualBlock(in_channels, out_channels, stride),
                                 *(ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1)))
        self.stage1 = stage(stem_channels, c1, 1, b1)
        self.exit1 = classifier(c1, dropouts[0], num_classes)
        self.stage2 = stage(c1, c2, 2, b2)
        self.exit2 = classifier(c2, dropouts[1], num_classes)
        self.stage3 = stage(c2, c3, 2, b3)
        self.final = classifier(c3, dropouts[2], num_classes)
        if final_mode == "residual":
            self.reset_final_correction()

    def reset_final_correction(self):
        """Start a residual final head with exactly the Exit 2 predictions."""
        nn.init.zeros_(self.final[-1].weight)
        nn.init.zeros_(self.final[-1].bias)

    def final_logits(self, features, exit2_logits):
        correction = self.final(features)
        return (exit2_logits.detach() + correction
                if self.config["final_mode"] == "residual" else correction)

    def forward(self, x):
        x = self.stage1(self.stem(x))
        logits1 = self.exit1(x)
        x = self.stage2(x)
        logits2 = self.exit2(x)
        x = self.stage3(x)
        return logits1, logits2, self.final_logits(x, logits2)

    @torch.inference_mode()
    def adaptive_forward(self, x, threshold_1, threshold_2, *, temperatures=(1., 1., 1.),
                         mode="normal") -> AdaptiveOutput:
        """Compact the remaining batch after each exit, preserving sample order.

        Returned logits are temperature-scaled. Requires eval() to make dropout
        and batch normalization independent of batch compaction.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before adaptive inference")
        if mode not in {"normal", "low-power"}:
            raise ValueError("mode must be normal or low-power")
        if not 0 <= threshold_1 <= 1 or not 0 <= threshold_2 <= 1:
            raise ValueError("Thresholds must lie in [0, 1]")
        if len(temperatures) != 3 or any(not 0 < float(t) < float("inf") for t in temperatures):
            raise ValueError("Three finite positive temperatures are required")
        if x.ndim != 3 or tuple(x.shape[1:]) != (9, 128) or not len(x):
            raise ValueError("Expected a nonempty (batch, 9, 128) tensor")
        logits = x.new_empty((len(x), self.config["num_classes"]))
        exits = torch.empty(len(x), dtype=torch.long, device=x.device)
        remaining = torch.arange(len(x), device=x.device)
        features = self.stage1(self.stem(x))
        first = self.exit1(features) / temperatures[0]
        leave = first.softmax(dim=1).amax(dim=1) >= threshold_1
        logits[remaining[leave]], exits[remaining[leave]] = first[leave], 1
        remaining, features = remaining[~leave], features[~leave]
        if len(remaining):
            features = self.stage2(features)
            second_logits = self.exit2(features)
            second = second_logits / temperatures[1]
            leave = (torch.ones(len(remaining), dtype=torch.bool, device=x.device) if mode == "low-power"
                     else second.softmax(dim=1).amax(dim=1) >= threshold_2)
            logits[remaining[leave]], exits[remaining[leave]] = second[leave], 2
            remaining, features = remaining[~leave], features[~leave]
            if len(remaining):
                logits[remaining] = self.final_logits(self.stage3(features), second_logits[~leave]) / temperatures[2]
                exits[remaining] = 3
        confidence, predictions = logits.softmax(dim=1).max(dim=1)
        return AdaptiveOutput(logits, predictions, confidence, exits)


def model_summary(model):
    """Count Conv/Linear MACs on one window; omit BN/ReLU/pool/add operations."""
    was_training = model.training
    model.eval()
    macs, shapes, handles = {}, {}, []
    def hook(name):
        def record(module, inputs, output):
            if isinstance(module, nn.Conv1d):
                macs[name] = output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
            elif isinstance(module, nn.Linear):
                macs[name] = output.numel() * module.in_features
        return record
    def shape_hook(name):
        return lambda module, inputs, output: shapes.__setitem__(name, list(output.shape))
    try:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                handles.append(module.register_forward_hook(hook(name)))
        for name in ["stem", "stage1", "exit1", "stage2", "exit2", "stage3", "final"]:
            handles.append(getattr(model, name).register_forward_hook(shape_hook(name)))
        with torch.inference_mode():
            model(torch.zeros(1, 9, 128, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    exits = []
    prefixes = []
    for names in [("stem", "stage1", "exit1"), ("stage2", "exit2"), ("stage3", "final")]:
        prefixes.extend(names)
        params = sum(p.numel() for name, p in model.named_parameters() if name.split(".")[0] in prefixes)
        executed_macs = sum(v for name, v in macs.items() if name.split(".")[0] in prefixes)
        exits.append({"parameters_up_to_exit": params, "macs_per_window": executed_macs,
                      "approx_flops_per_window": 2 * executed_macs})
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "model_size": model.config["model_size"],
            "final_mode": model.config["final_mode"],
            "channels": list(MODEL_SIZES[model.config["model_size"]]),
            "stem_channels": model.stem[0].out_channels,
            "stage_blocks": list(STAGE_BLOCKS[model.config["model_size"]]),
            "parameter_bytes_float32": 4 * sum(p.numel() for p in model.parameters()),
            "input_shape": [1, 9, 128], "shapes": shapes, "exits": exits,
            "computation_note": "Conv1d/Linear MACs only, including earlier exit heads; FLOPs = 2 * MACs. Not measured latency or energy."}
