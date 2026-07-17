from types import SimpleNamespace

from omegaconf import OmegaConf

from scaletraining.util import wandb_utils
from scaletraining.util.device import resolve_device


class FakeRun:
    def __init__(self, *, mode="online", url="https://wandb.ai/entity/project/runs/abc123"):
        self.settings = SimpleNamespace(mode=mode)
        self.entity = "entity"
        self.project = "project"
        self.id = "abc123"
        self.path = (self.entity, self.project, self.id)
        self.url = url
        self.summary = {}


class FakeWandb:
    def __init__(
        self,
        *,
        mode="online",
        run_url="https://wandb.ai/entity/project/runs/abc123",
        init_error=None,
    ):
        self.run = None
        self.mode = mode
        self.run_url = run_url
        self.init_error = init_error
        self.init_kwargs = None
        self.metric_definitions = []
        self.logged = []
        self.finished = False
        self.finish_exit_code = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        if self.init_error:
            raise self.init_error
        self.run = FakeRun(mode=self.mode, url=self.run_url)
        return self.run

    def define_metric(self, name, **kwargs):
        self.metric_definitions.append((name, kwargs))

    def log(self, payload):
        self.logged.append(payload)

    def finish(self, *, exit_code):
        self.finished = True
        self.finish_exit_code = exit_code
        self.run = None


def _cfg():
    return OmegaConf.create(
        {
            "tokenizer": {
                "tokenizer_name": "fixture-tokenizer",
                "pretrained_tokenizer_name": "fixture-tokenizer",
            },
            "logging": {
                "wandb_project_name": "project",
                "experiment_tags": ["smoke"],
            },
            "sweep": None,
        }
    )


def test_init_defines_schema_v1_metrics_and_returns_serializable_identity(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setattr(wandb_utils, "wandb_sdk", fake)

    identity = wandb_utils.init_wandb(
        _cfg(),
        tokenizer_vocab_size=128,
        tok=SimpleNamespace(name_or_path="fixture-tokenizer"),
    )

    assert identity.to_dict() == {
        "provider": "wandb",
        "schema_version": 1,
        "state": "initialized",
        "mode": "online",
        "entity": "entity",
        "project": "project",
        "run_id": "abc123",
        "path": "entity/project/abc123",
        "url": "https://wandb.ai/entity/project/runs/abc123",
        "error": None,
    }
    assert fake.init_kwargs["config"]["tracking_schema_version"] == 1
    assert fake.init_kwargs["tags"] == ["smoke"]
    assert fake.metric_definitions == [
        ("progress/tokens", {}),
        ("progress/optimizer_step", {}),
        ("train/*", {"step_metric": "progress/tokens"}),
        ("validation/*", {"step_metric": "progress/tokens"}),
        ("performance/*", {"step_metric": "progress/tokens"}),
        ("compute/*", {"step_metric": "progress/tokens"}),
        ("moe/*", {"step_metric": "progress/tokens"}),
    ]


def test_cuda_fallback_provenance_is_in_wandb_config(monkeypatch):
    cfg = _cfg()
    cfg.device = {"device": "cuda"}
    cfg.device_requested = None
    cfg.device_resolved = None
    fake = FakeWandb()
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(wandb_utils, "wandb_sdk", fake)

    assert resolve_device(cfg) == "cpu"
    wandb_utils.init_wandb(
        cfg,
        tokenizer_vocab_size=128,
        tok=SimpleNamespace(name_or_path="fixture-tokenizer"),
    )

    config = fake.init_kwargs["config"]
    assert config["device_requested"] == "cuda"
    assert config["device_resolved"] == "cpu"
    assert config["device"]["device"] == "cpu"


def test_schema_v1_payloads_share_explicit_token_and_optimizer_axes(monkeypatch):
    fake = FakeWandb()
    fake.run = FakeRun()
    monkeypatch.setattr(wandb_utils, "wandb_sdk", fake)

    wandb_utils.log_train_metrics(
        used_tokens=64,
        optimizer_step=2,
        loss=1.25,
        lr=0.001,
        grad_norm_pre_clip=3.5,
        throughput=128.0,
        flops_used=2048.0,
        peak_memory_allocated_bytes=1024,
        peak_memory_reserved_bytes=2048,
    )
    wandb_utils.log_eval_metrics(
        used_tokens=64,
        optimizer_step=2,
        val_loss=1.5,
        val_perplexity=4.48,
    )
    wandb_utils.log_moe_metrics(
        used_tokens=64,
        optimizer_step=2,
        metrics={"moe/load_cv_mean": 0.2},
    )

    progress = {"progress/tokens": 64, "progress/optimizer_step": 2}
    assert fake.logged[0] == {
        **progress,
        "train/loss_per_token": 1.25,
        "train/learning_rate": 0.001,
        "train/grad_norm_pre_clip": 3.5,
        "performance/tokens_per_second": 128.0,
        "compute/flops_total": 2048.0,
        "compute/peak_memory_allocated_bytes": 1024,
        "compute/peak_memory_reserved_bytes": 2048,
    }
    assert fake.logged[1] == {
        **progress,
        "validation/loss_per_token": 1.5,
        "validation/perplexity": 4.48,
    }
    assert fake.logged[2] == {**progress, "moe/load_cv_mean": 0.2}


def test_disabled_unavailable_and_initialization_failure_states(monkeypatch):
    disabled = FakeWandb(mode="disabled")
    monkeypatch.setattr(wandb_utils, "wandb_sdk", disabled)
    identity = wandb_utils.init_wandb(
        _cfg(), 128, SimpleNamespace(name_or_path="fixture-tokenizer")
    )
    assert identity.state == "disabled"
    assert identity.mode == "disabled"
    assert identity.path is None
    assert identity.url is None

    offline = FakeWandb(mode="offline", run_url=None)
    monkeypatch.setattr(wandb_utils, "wandb_sdk", offline)
    identity = wandb_utils.init_wandb(
        _cfg(), 128, SimpleNamespace(name_or_path="fixture-tokenizer")
    )
    assert identity.state == "initialized"
    assert identity.mode == "offline"
    assert identity.path == "entity/project/abc123"
    assert identity.url is None

    monkeypatch.setattr(wandb_utils, "wandb_sdk", None)
    assert wandb_utils.init_wandb(_cfg(), 128, object()).state == "unavailable"

    failed = FakeWandb(init_error=RuntimeError("network down"))
    monkeypatch.setattr(wandb_utils, "wandb_sdk", failed)
    identity = wandb_utils.init_wandb(_cfg(), 128, object())
    assert identity.state == "initialization_failed"
    assert identity.error == "network down"


def test_finish_is_a_noop_without_a_run_and_finishes_an_active_run(monkeypatch):
    fake = FakeWandb()
    monkeypatch.setattr(wandb_utils, "wandb_sdk", fake)
    wandb_utils.finish_wandb(exit_code=0)
    assert fake.finished is False

    fake.run = FakeRun()
    wandb_utils.finish_wandb(exit_code=7)
    assert fake.finished is True
    assert fake.finish_exit_code == 7
