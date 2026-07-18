import contextlib

import torch

from scaletraining.util import generation_utils


class DeviceIgnoringTensor:
    def __init__(self, value):
        self.value = value

    def to(self, device):
        return self.value


class TinyTokenizer:
    eos_token_id = 7
    pad_token_id = 7
    eos_token = "<eos>"
    pad_token = "<eos>"

    def encode(self, prompt, return_tensors):
        return DeviceIgnoringTensor(torch.tensor([[1, 2]]))

    def decode(self, input_ids, skip_special_tokens):
        return "decoded"


class TinyGenerationModel(torch.nn.Module):
    def forward(self, input_ids):
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)


def test_generation_uses_autocast_for_indexed_cuda(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def fake_autocast(**kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr(generation_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generation_utils, "autocast", fake_autocast)

    result = generation_utils.generate_autoregressive(
        TinyGenerationModel(),
        TinyTokenizer(),
        "cuda:1",
        prompt="hello",
        max_new_tokens=1,
    )

    assert result == "decoded"
    assert calls == [{"device_type": "cuda", "dtype": torch.bfloat16}]
