import torch
from omegaconf import OmegaConf

from scaletraining.model import training_loop


class TinyModel(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.events = events

    def forward_hidden(self, input_ids):
        self.events.append("forward")
        batch, tokens = input_ids.shape
        return self.weight * torch.ones(batch, tokens, 1)


class RecordingOptimizer:
    def __init__(self, parameter, events):
        self.parameter = parameter
        self.events = events
        self.param_groups = [{"lr": 0.01}]

    def step(self):
        self.events.append("optimizer_step")

    def zero_grad(self, *, set_to_none):
        if set_to_none:
            self.parameter.grad = None


def _cfg():
    return OmegaConf.create(
        {
            "device": {"device": "cpu"},
            "device_resolved": "cpu",
            "logging": {
                "log_implementation_details": False,
                "debug_memory": False,
            },
            "training": {
                "max_train_tokens": 4,
                "accum_steps": 2,
                "grad_clip_norm": 1.0,
                "logits_chunk_size": 0,
                "early_stop_tokens_without_improvement": 0,
                "early_stop_min_delta": 0.0,
                "eval_interval_tokens": 0,
                "eval_max_batches": 0,
            },
            "optimizer": {},
            "model": {
                "n_embed": 1,
                "n_hidden": 1,
                "n_head": 1,
                "max_seq_len": 2,
                "n_layer": 1,
            },
            "moe": {
                "use_moe": False,
                "moe_n_layers": 0,
                "moe_top_k": 1,
                "moe_n_experts": 1,
                "moe_lb_coef": 0.0,
            },
        }
    )


def test_accumulation_timing_and_gradient_norm_cover_the_complete_window(monkeypatch):
    events = []
    captured = []
    model = TinyModel(events)
    optimizer = RecordingOptimizer(model.weight, events)
    clock = iter([10.0, 12.0])

    monkeypatch.setattr(
        training_loop,
        "split_model_matrix_params",
        lambda *args, **kwargs: ([model.weight], []),
    )
    monkeypatch.setattr(
        training_loop,
        "build_optimizers",
        lambda *args, **kwargs: (optimizer, None),
    )
    monkeypatch.setattr(training_loop, "scale_optimizer_lr", lambda *args: None)
    monkeypatch.setattr(training_loop, "compute_lr_scale_tokens", lambda *args: 1.0)
    monkeypatch.setattr(training_loop, "compute_progress_t", lambda *args: 0.0)
    monkeypatch.setattr(
        training_loop,
        "apply_moe_schedules",
        lambda model, moe_cfg, progress: moe_cfg.moe_lb_coef,
    )
    monkeypatch.setattr(training_loop, "estimate_flops", lambda **kwargs: 100.0)
    monkeypatch.setattr(
        training_loop,
        "prepare_targets",
        lambda input_ids: (torch.zeros_like(input_ids), input_ids.numel()),
    )
    monkeypatch.setattr(
        training_loop,
        "compute_loss_sum",
        lambda model, hidden, targets, chunk_size, loss_fn: hidden.sum(),
    )

    def clip(*args, **kwargs):
        events.append("clip")
        return torch.tensor(7.5)

    def perf_counter():
        events.append("clock")
        return next(clock)

    monkeypatch.setattr(training_loop.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(training_loop.time, "perf_counter", perf_counter)
    monkeypatch.setattr(
        training_loop,
        "log_train_metrics",
        lambda **kwargs: captured.append(kwargs),
    )

    batches = [
        {"input_ids": torch.tensor([[1, 2]])},
        {"input_ids": torch.tensor([[3, 4]])},
    ]
    stats = training_loop.training_run(
        _cfg(),
        model,
        batches,
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
    )

    assert events == [
        "clock",
        "forward",
        "forward",
        "clip",
        "optimizer_step",
        "clock",
    ]
    assert stats == {"train_loss": [0.5]}
    assert captured == [
        {
            "used_tokens": 4,
            "optimizer_step": 1,
            "loss": 0.5,
            "lr": 0.01,
            "grad_norm_pre_clip": 7.5,
            "throughput": 2.0,
            "flops_used": 100.0,
            "peak_memory_allocated_bytes": None,
            "peak_memory_reserved_bytes": None,
        }
    ]


def test_timing_synchronization_only_runs_for_available_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(training_loop.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        training_loop.torch.cuda,
        "synchronize",
        lambda *, device: calls.append(device),
    )

    training_loop._synchronize_for_timing(torch.device("cpu"))
    training_loop._synchronize_for_timing(torch.device("cuda:1"))

    assert calls == [torch.device("cuda:1")]


def test_peak_memory_operations_use_the_selected_cuda_device(monkeypatch):
    calls = []
    device = torch.device("cuda:1")
    monkeypatch.setattr(training_loop.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        training_loop.torch.cuda,
        "reset_peak_memory_stats",
        lambda *, device: calls.append(("reset", device)),
    )
    monkeypatch.setattr(
        training_loop.torch.cuda,
        "max_memory_allocated",
        lambda *, device: calls.append(("allocated", device)) or 1024,
    )
    monkeypatch.setattr(
        training_loop.torch.cuda,
        "max_memory_reserved",
        lambda *, device: calls.append(("reserved", device)) or 2048,
    )

    training_loop._reset_peak_memory_stats(device)
    assert training_loop._peak_memory_stats(device) == (1024, 2048)
    assert training_loop._peak_memory_stats(torch.device("cpu")) == (None, None)

    assert calls == [
        ("reset", device),
        ("allocated", device),
        ("reserved", device),
    ]
