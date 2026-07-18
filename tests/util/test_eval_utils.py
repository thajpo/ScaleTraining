import contextlib
import json
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from scaletraining.util import eval_utils
from scaletraining.util.eval_utils import (
    evaluate_perplexity,
    evaluate_perplexity_stats,
    write_eval_result,
    write_lm_eval_result,
)


class TinyEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.W_ue = nn.Linear(4, 8)

    def forward_hidden(self, input_ids):
        return self.embedding(input_ids)

    def forward(self, input_ids):
        return self.W_ue(self.forward_hidden(input_ids))


def _cfg(tmp_path):
    checkpoint = tmp_path / "run" / "model.pt"
    return SimpleNamespace(
        device=SimpleNamespace(device="cpu"),
        device_resolved=None,
        generation=SimpleNamespace(model_path=str(checkpoint)),
        eval=SimpleNamespace(tasks="hellaswag", write_results=True, output_dir=None),
        tokenizer=SimpleNamespace(
            dataset_names=["test/dataset"],
            dataset_tag=[None],
            tokenizer_name="test-tokenizer",
            pretrained_tokenizer_name="test-tokenizer",
            hf_dataset_config_name=None,
        ),
        model=SimpleNamespace(
            n_layer=1,
            n_head=1,
            n_embed=4,
            n_hidden=8,
            max_seq_len=8,
        ),
        moe=SimpleNamespace(use_moe=False, moe_n_layers=0),
        training=SimpleNamespace(
            seed=13,
            batch_size=1,
            accum_steps=1,
            max_train_tokens=8,
            eval_batch_size=1,
            eval_max_batches=1,
            logits_chunk_size=0,
        ),
        optimizer=SimpleNamespace(primary_optimizer="adamw", lr=0.001),
    )


def _loader():
    return DataLoader([{"input_ids": torch.tensor([1, 2, 3])}], batch_size=1)


def test_evaluate_perplexity_keeps_two_tuple_return(tmp_path):
    cfg = _cfg(tmp_path)
    model = TinyEvalModel()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    result = evaluate_perplexity(model, _loader(), cfg, loss_fn, max_batches=1)
    stats = evaluate_perplexity_stats(model, _loader(), cfg, loss_fn, max_batches=1)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert stats["tokens"] == 2
    assert stats["batches"] == 1


def test_evaluate_perplexity_uses_autocast_for_indexed_cuda(
    tmp_path, monkeypatch
):
    calls = []
    cfg = _cfg(tmp_path)
    model = TinyEvalModel()

    class DeviceIgnoringTensor:
        def to(self, device):
            calls.append(("to", device))
            return torch.tensor([[1, 2, 3]])

    @contextlib.contextmanager
    def fake_autocast(**kwargs):
        calls.append(("autocast", kwargs))
        yield

    monkeypatch.setattr(eval_utils, "resolve_device", lambda cfg: "cuda:1")
    monkeypatch.setattr(eval_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(eval_utils, "autocast", fake_autocast)

    stats = evaluate_perplexity_stats(
        model,
        [{"input_ids": DeviceIgnoringTensor()}],
        cfg,
        nn.CrossEntropyLoss(reduction="sum"),
        max_batches=1,
    )

    assert stats["tokens"] == 2
    assert calls == [
        ("to", "cuda:1"),
        ("autocast", {"device_type": "cuda", "dtype": torch.bfloat16}),
    ]


def test_write_eval_result_defaults_next_to_checkpoint(tmp_path):
    cfg = _cfg(tmp_path)
    metrics = {
        "loss": 1.25,
        "perplexity": 3.49,
        "total_loss": 2.5,
        "tokens": 2,
        "batches": 1,
        "max_batches": 1,
    }

    result_path = write_eval_result(cfg, metrics)
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert result_path == tmp_path / "run" / "eval_results.json"
    assert payload["schema_version"] == 1
    assert payload["checkpoint"]["path"] == str(tmp_path / "run" / "model.pt")
    assert payload["dataset"]["fingerprint_short"]
    assert payload["validation"]["tokens"] == 2
    assert payload["config_summary"]["training"]["seed"] == 13


def test_write_lm_eval_result_preserves_tasks_and_results(tmp_path):
    cfg = _cfg(tmp_path)

    result_path = write_lm_eval_result(
        cfg,
        ["hellaswag"],
        {"results": {"hellaswag": {"acc,none": 0.125}}},
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert result_path == tmp_path / "run" / "lm_eval_results.json"
    assert payload["tasks"] == ["hellaswag"]
    assert payload["results"]["results"]["hellaswag"]["acc,none"] == 0.125
