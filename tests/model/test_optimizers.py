import torch
import torch.nn as nn
import pytest

from scaletraining.model.optimizers import Muon, AdaMuon


def get_device():
    """Get the best available device (CUDA/ROCm > CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "hip") and torch.hip.is_available():
        return torch.device("hip")
    else:
        return torch.device("cpu")


class SimpleNet(nn.Module):
    """Tiny MLP with no biases so all params are 2D matrices.
    This avoids shape issues in custom optimizers that assume matrices.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden, bias=False)
        self.l2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = torch.relu_(x)
        return self.l2(x)


def _make_teacher_data(
    n: int = 256, d_in: int = 16, d_hidden: int = 32, d_out: int = 16
):
    device = get_device()
    torch.manual_seed(42)
    X = torch.randn(n, d_in, device=device)
    # Create a fixed teacher network to generate targets deterministically
    teacher = SimpleNet(d_in, d_hidden, d_out).to(device)
    for p in teacher.parameters():
        nn.init.normal_(p, mean=0.0, std=0.5)
    with torch.no_grad():
        Y = teacher(X)
    return X, Y, d_in, d_hidden, d_out


def _train_and_record_losses(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    steps: int = 60,
):
    crit = nn.MSELoss()
    losses = []
    for _ in range(steps):
        preds = model(X)
        loss = crit(preds, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def test_optimizers_smoke():
    device = get_device()
    torch.manual_seed(123)

    X, Y, d_in, d_hidden, d_out = _make_teacher_data(n=64, d_in=8, d_hidden=16, d_out=8)
    lr = 1e-3
    steps = 200

    base = SimpleNet(d_in, d_hidden, d_out).to(device)
    base_opt = torch.optim.AdamW(
        base.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0
    )
    base_losses = _train_and_record_losses(base, base_opt, X, Y, steps)
    assert base_losses[-1] < base_losses[0], "AdamW baseline did not decrease loss"

    for OptCls in (Muon, AdaMuon):
        torch.manual_seed(123)
        model = SimpleNet(d_in, d_hidden, d_out).to(device)
        opt = OptCls(
            model.parameters(),
            lr=lr,
            beta=0.9,
            beta2=0.999,
            weight_decay=0.0,
            ns_iters=3,
            eps=1e-8,
        )
        losses = _train_and_record_losses(model, opt, X, Y, steps)
        assert losses[-1] < losses[0], f"{OptCls.__name__} did not decrease loss"


@pytest.mark.slow
def test_optimizers_match_adamw_baseline():
    device = get_device()
    print(f"Using device: {device}")

    # Fixed seed for determinism
    torch.manual_seed(123)

    X, Y, d_in, d_hidden, d_out = _make_teacher_data()

    # Common hyperparams kept small and stable
    lr = 5e-4
    steps = 60000

    # AdamW baseline
    base = SimpleNet(d_in, d_hidden, d_out).to(device)
    base_opt = torch.optim.AdamW(
        base.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0
    )
    base_losses = _train_and_record_losses(base, base_opt, X, Y, steps)

    # Sanity: baseline should decrease
    assert base_losses[-1] < base_losses[0], "AdamW baseline did not decrease loss"

    # Compare Muon and AdaMuon against AdamW
    for OptCls in (Muon, AdaMuon):
        torch.manual_seed(123)  # reset init for fair comparison
        model = SimpleNet(d_in, d_hidden, d_out).to(device)
        opt = OptCls(
            model.parameters(),
            lr=lr,
            beta=0.9,
            beta2=0.999,
            weight_decay=0.0,
            ns_iters=3,
            eps=1e-8,
        )
        losses = _train_and_record_losses(model, opt, X, Y, steps)

        # Each should also decrease
        assert losses[-1] < losses[0], f"{OptCls.__name__} did not decrease loss"

        # Final loss should be within a reasonable factor of AdamW
        ratio = losses[-1] / (base_losses[-1] + 1e-12)
        assert ratio < 1.25, (
            f"{OptCls.__name__} final loss {losses[-1]:.4f} not similar to AdamW {base_losses[-1]:.4f}"
        )
        print(f"Adam loss: {base_losses[-1]}, Other loss {losses[-1]}")

        # And trend over the last segment should be broadly similar
        tail = 10
        mu = sum(losses[-tail:]) / tail
        mu_base = sum(base_losses[-tail:]) / tail
        assert mu <= 1.25 * mu_base, (
            f"{OptCls.__name__} trailing loss worse than 3x AdamW"
        )


def main():
    test_optimizers_match_adamw_baseline()


if __name__ == "__main__":
    main()
