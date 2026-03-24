"""Probabilistic and behavioral proxy models used in the report."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence

from .common import BIN_DIGITS, END, START, Language, bits_for_max_value, encode_int_bin, kl_tv_over_shared_support, normalize_distribution
from .metrics import TextBigramMetrics


class CharBigramLM:
    """Simple add-alpha character bigram language model."""

    def __init__(self, alpha: float = 1.0, vocab: str = BIN_DIGITS) -> None:
        self.alpha = alpha
        self.vocab = set(vocab) | {START, END}
        self.counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.context_totals: Counter[str] = Counter()

    def update(self, text: str) -> None:
        tagged = START + text + END
        for index in range(len(tagged) - 1):
            left = tagged[index]
            right = tagged[index + 1]
            self.counts[left][right] += 1
            self.context_totals[left] += 1

    def train(self, data: Sequence[str]) -> None:
        for text in data:
            self.update(text)

    def prob_string(self, text: str) -> float:
        tagged = START + text + END
        vocab_size = len(self.vocab)
        log_prob = 0.0
        for index in range(len(tagged) - 1):
            left = tagged[index]
            right = tagged[index + 1]
            numerator = self.counts[left][right] + self.alpha
            denominator = self.context_totals[left] + self.alpha * vocab_size
            log_prob += math.log(numerator / denominator)
        return math.exp(log_prob)


def bigram_metrics_on_text(
    stream: Sequence[int],
    target: Language,
    support_max: int = 5000,
) -> TextBigramMetrics:
    """Fit a bigram LM on the observed text and compare it to uniform-on-target."""
    bits = bits_for_max_value(support_max)
    lm = CharBigramLM(alpha=1.0, vocab=BIN_DIGITS)
    # Fixed-width binary encoding prevents the discriminator and LM from leaning
    # too heavily on trivial string-length cues.
    lm.train([encode_int_bin(value, bits) for value in stream])

    support_values = [value for value in range(1, support_max + 1) if target(value)]
    support = [encode_int_bin(value, bits) for value in support_values]
    if not support:
        return TextBigramMetrics(kl=float("nan"), tv=float("nan"), support_size=0)

    # The comparison target is deliberately simple: uniform mass over the true
    # language restricted to the finite support window.
    uniform_target = {token: 1.0 / len(support) for token in support}
    model_scores = {token: lm.prob_string(token) for token in support}
    normalized_scores = normalize_distribution(model_scores)
    kl, tv = kl_tv_over_shared_support(uniform_target, normalized_scores)
    return TextBigramMetrics(kl=kl, tv=tv, support_size=len(support))


def bigram_features(text: str, bigram_vocab: Sequence[str]) -> list[float]:
    """Count character bigrams for a fixed vocabulary."""
    tagged = START + text + END
    counts = Counter(tagged[index : index + 2] for index in range(len(tagged) - 1))
    return [float(counts[bigram]) for bigram in bigram_vocab]


def sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid."""
    if value >= 0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def train_logreg(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    lr: float = 0.2,
    epochs: int = 80,
    l2: float = 1e-4,
    seed: int = 0,
) -> list[float]:
    """Train a tiny logistic regression model with SGD."""
    if not features:
        raise ValueError("features must be non-empty")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")

    rng = random.Random(seed)
    num_examples = len(features)
    num_features = len(features[0])
    weights = [0.0] * (num_features + 1)  # bias term at the end

    for _ in range(epochs):
        # Plain SGD is enough here because this classifier is only a lightweight
        # behavioral proxy, not a production model.
        indices = list(range(num_examples))
        rng.shuffle(indices)
        for example_index in indices:
            row = features[example_index]
            label = labels[example_index]
            score = sum(weights[j] * row[j] for j in range(num_features)) + weights[num_features]
            prob = sigmoid(score)
            error = prob - label
            for j in range(num_features):
                weights[j] -= lr * (error * row[j] + l2 * weights[j])
            weights[num_features] -= lr * error

    return weights


def eval_logreg(features: Sequence[Sequence[float]], labels: Sequence[int], weights: Sequence[float]) -> float:
    """Evaluate a logistic regression classifier by accuracy."""
    if not features:
        raise ValueError("features must be non-empty")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")

    num_features = len(features[0])
    correct = 0
    for row, label in zip(features, labels):
        score = sum(weights[j] * row[j] for j in range(num_features)) + weights[num_features]
        prediction = 1 if sigmoid(score) >= 0.5 else 0
        correct += int(prediction == label)
    return correct / len(labels)


def sample_uniform_language(target: Language, support_max: int, sample_size: int, seed: int = 0) -> list[int]:
    """Sample with replacement from the target language restricted to ``[1, support_max]``."""
    rng = random.Random(seed)
    pool = [value for value in range(1, support_max + 1) if target(value)]
    if not pool:
        return []
    return [rng.choice(pool) for _ in range(sample_size)]


def discriminator_accuracy_bin(
    real_values: Sequence[int],
    generated_values: Sequence[int],
    support_max: int,
    seed: int = 0,
) -> float:
    """Run a simple bigram-feature logistic discriminator."""
    if not real_values or not generated_values:
        raise ValueError("real_values and generated_values must be non-empty")

    bits = bits_for_max_value(support_max)
    chars = [START, END, *BIN_DIGITS]
    # Character bigrams keep the feature space tiny and reproducible while still
    # being sensitive to coarse structural differences in the generated strings.
    bigram_vocab = [left + right for left in chars for right in chars]

    real_tokens = [encode_int_bin(value, bits) for value in real_values]
    generated_tokens = [encode_int_bin(value, bits) for value in generated_values]

    features = [bigram_features(token, bigram_vocab) for token in real_tokens]
    features += [bigram_features(token, bigram_vocab) for token in generated_tokens]
    labels = [1] * len(real_tokens) + [0] * len(generated_tokens)

    rng = random.Random(seed)
    indices = list(range(len(features)))
    rng.shuffle(indices)
    # We keep a simple random split because the goal is relative separability,
    # not a precise generalization estimate.
    split = int(0.7 * len(indices))
    split = max(1, min(split, len(indices) - 1))

    train_indices = indices[:split]
    test_indices = indices[split:]

    train_features = [features[index] for index in train_indices]
    train_labels = [labels[index] for index in train_indices]
    test_features = [features[index] for index in test_indices]
    test_labels = [labels[index] for index in test_indices]

    weights = train_logreg(train_features, train_labels, lr=0.2, epochs=80, l2=1e-4, seed=seed)
    return eval_logreg(test_features, test_labels, weights)
