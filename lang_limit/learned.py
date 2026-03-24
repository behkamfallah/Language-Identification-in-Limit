"""A small learned autoregressive bridge experiment without membership oracles."""

from __future__ import annotations

from dataclasses import dataclass

from .common import AdversaryMode, Language
from .metrics import DiscriminatorMetrics
from .models import discriminator_accuracy_bin, sample_uniform_language

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


@dataclass(frozen=True)
class LearnedModelConfig:
    """Configuration for the tiny autoregressive bridge model."""

    context_len: int = 8
    embedding_dim: int = 16
    hidden_dim: int = 32
    num_layers: int = 1
    epochs: int = 300
    learning_rate: float = 0.02
    temperature: float = 0.8
    num_rollouts: int = 16
    rollout_steps: int = 40
    discriminator_min_sample: int = 30


@dataclass(frozen=True)
class LearnedModelResult:
    """Summary of the learned autoregressive bridge experiment."""

    mode: AdversaryMode
    seed: int
    valid_rate: float
    novelty_rate: float
    coverage_ratio: float
    unique_valid_outputs: int
    distinct_rollout_ratio: float
    discriminator: DiscriminatorMetrics
    example_rollout: tuple[int, ...]


class _NextDeltaGRU(nn.Module):
    """Tiny GRU that predicts the next integer-difference token."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        states, _ = self.gru(embedded)
        return self.output(states[:, -1, :])


def _require_torch() -> None:
    """Fail with a clear message when the optional torch dependency is missing."""
    if torch is None or nn is None:
        raise RuntimeError(
            "The learned autoregressive bridge requires PyTorch. "
            "Install torch or skip the learned-model experiment."
        )


def _difference_stream(stream: list[int]) -> list[int]:
    """Convert a stream of integers into first differences."""
    return [right - left for left, right in zip(stream, stream[1:])]


def _build_training_windows(
    differences: list[int],
    context_len: int,
) -> tuple[list[list[int]], list[int], dict[int, int], dict[int, int]]:
    """Create sliding windows for next-token prediction over difference tokens."""
    vocab = sorted(set(differences))
    stoi = {value: index for index, value in enumerate(vocab)}
    itos = {index: value for value, index in stoi.items()}

    features: list[list[int]] = []
    labels: list[int] = []
    for start in range(len(differences) - context_len):
        features.append([stoi[value] for value in differences[start : start + context_len]])
        labels.append(stoi[differences[start + context_len]])

    return features, labels, stoi, itos


def _train_model(
    features: list[list[int]],
    labels: list[int],
    vocab_size: int,
    config: LearnedModelConfig,
    seed: int,
) -> _NextDeltaGRU:
    """Train the small GRU on the observed difference sequence."""
    _require_torch()
    if not features:
        raise ValueError("not enough data to build autoregressive training windows")

    torch.manual_seed(seed)
    inputs = torch.tensor(features, dtype=torch.long)
    targets = torch.tensor(labels, dtype=torch.long)

    model = _NextDeltaGRU(
        vocab_size,
        config.embedding_dim,
        config.hidden_dim,
        config.num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(config.epochs):
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def _sample_rollouts(
    model: _NextDeltaGRU,
    stream: list[int],
    differences: list[int],
    stoi: dict[int, int],
    itos: dict[int, int],
    config: LearnedModelConfig,
    seed: int,
) -> list[list[int]]:
    """Roll the trained GRU forward from the last observed context."""
    _require_torch()
    context = [stoi[value] for value in differences[-config.context_len :]]
    rollouts: list[list[int]] = []

    for rollout_index in range(config.num_rollouts):
        current_value = stream[-1]
        current_context = context[:]
        generated: list[int] = []
        generator = torch.Generator().manual_seed(seed + rollout_index)

        for _ in range(config.rollout_steps):
            logits = model(torch.tensor([current_context], dtype=torch.long))[0]
            probs = torch.softmax(logits / config.temperature, dim=0)
            token = torch.multinomial(probs, 1, generator=generator).item()
            current_value += itos[token]
            generated.append(current_value)
            current_context = current_context[1:] + [token]

        rollouts.append(generated)

    return rollouts


def _evaluate_rollouts(
    mode: AdversaryMode,
    seed: int,
    target: Language,
    train_stream: list[int],
    rollouts: list[list[int]],
    support_max: int,
    min_discriminator_sample: int,
) -> LearnedModelResult:
    """Collapse rollout samples into validity, coverage, and collapse metrics."""
    train_seen = set(train_stream)
    flat_outputs = [value for rollout in rollouts for value in rollout]
    valid_outputs = [value for value in flat_outputs if target(value)]
    clipped_valid_outputs = [value for value in valid_outputs if value <= support_max]
    support_size = sum(1 for value in range(1, support_max + 1) if target(value))

    valid_rate = sum(target(value) for value in flat_outputs) / len(flat_outputs)
    novelty_rate = sum(value not in train_seen for value in flat_outputs) / len(flat_outputs)
    unique_valid_outputs = len(set(clipped_valid_outputs))
    coverage_ratio = unique_valid_outputs / support_size if support_size else 0.0
    distinct_rollout_ratio = len({tuple(rollout) for rollout in rollouts}) / len(rollouts)

    if len(clipped_valid_outputs) < min_discriminator_sample:
        discriminator = DiscriminatorMetrics(
            accuracy=None,
            support_max=support_max,
            sample_size=len(clipped_valid_outputs),
        )
    else:
        reference = sample_uniform_language(target, support_max, len(clipped_valid_outputs), seed=seed + 10_000)
        accuracy = discriminator_accuracy_bin(reference, clipped_valid_outputs, support_max, seed=seed + 20_000)
        discriminator = DiscriminatorMetrics(
            accuracy=accuracy,
            support_max=support_max,
            sample_size=len(clipped_valid_outputs),
        )

    return LearnedModelResult(
        mode=mode,
        seed=seed,
        valid_rate=valid_rate,
        novelty_rate=novelty_rate,
        coverage_ratio=coverage_ratio,
        unique_valid_outputs=unique_valid_outputs,
        distinct_rollout_ratio=distinct_rollout_ratio,
        discriminator=discriminator,
        example_rollout=tuple(rollouts[0][: min(10, len(rollouts[0]))]) if rollouts else tuple(),
    )


def run_learned_model_experiment(
    mode: AdversaryMode,
    seed: int,
    stream: list[int],
    target: Language,
    support_max: int,
    config: LearnedModelConfig | None = None,
) -> LearnedModelResult:
    """Train the small GRU bridge model and evaluate its generated rollouts."""
    config = config or LearnedModelConfig()
    if len(stream) < config.context_len + 2:
        raise ValueError("stream is too short for the learned-model experiment")

    differences = _difference_stream(stream)
    features, labels, stoi, itos = _build_training_windows(differences, config.context_len)
    model = _train_model(features, labels, vocab_size=len(stoi), config=config, seed=seed)
    rollouts = _sample_rollouts(model, stream, differences, stoi, itos, config=config, seed=seed)
    return _evaluate_rollouts(
        mode=mode,
        seed=seed,
        target=target,
        train_stream=stream,
        rollouts=rollouts,
        support_max=support_max,
        min_discriminator_sample=config.discriminator_min_sample,
    )


def format_learned_model_report(result: LearnedModelResult) -> str:
    """Render a short text summary for the learned bridge experiment."""
    accuracy = result.discriminator.accuracy
    accuracy_text = f"{accuracy:.3f}" if accuracy is not None else "NA"
    return (
        f"learned_gru (mode={result.mode}, seed={result.seed}): "
        f"valid_rate={result.valid_rate:.3f} "
        f"novelty_rate={result.novelty_rate:.3f} "
        f"coverage={result.coverage_ratio:.3f} "
        f"unique_valid={result.unique_valid_outputs} "
        f"distinct_rollouts={result.distinct_rollout_ratio:.3f} "
        f"disc_acc={accuracy_text} "
        f"example_rollout={list(result.example_rollout)}"
    )
