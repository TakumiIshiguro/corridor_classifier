import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from corridor_classifier.training import (
    configure_trainable_layers,
    create_optimizer,
    create_scheduler,
    last_blocks_for_epoch,
    learning_rate_multiplier,
    parameter_groups,
    run_epoch,
)


class FakeDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 8)


SCHEDULE = [
    {"epoch": 3, "last_blocks": 1},
    {"epoch": 5, "last_blocks": 2},
    {"epoch": 7, "last_blocks": 4},
]


def test_unfreeze_schedule_is_one_based():
    assert last_blocks_for_epoch(1, 2, SCHEDULE) == 0
    assert last_blocks_for_epoch(2, 2, SCHEDULE) == 0
    assert last_blocks_for_epoch(3, 2, SCHEDULE) == 1
    assert last_blocks_for_epoch(6, 2, SCHEDULE) == 2
    assert last_blocks_for_epoch(7, 2, SCHEDULE) == 4


def test_configure_trainable_layers_freezes_then_unfreezes():
    model = FakeDINO()
    configure_trainable_layers(model, 0)
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("head.")
    )

    configure_trainable_layers(model, 2)
    assert not any(
        parameter.requires_grad for parameter in model.blocks[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.blocks[-1].parameters()
    )
    assert all(parameter.requires_grad for parameter in model.norm.parameters())

    configure_trainable_layers(model, 4)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_optimizer_groups_include_frozen_parameters_for_later_unfreeze():
    model = FakeDINO()
    configure_trainable_layers(model, 0)
    groups = parameter_groups(model, 1e-3, 1e-5)

    backbone_count = sum(parameter.numel() for parameter in groups[0]["params"])
    head_count = sum(parameter.numel() for parameter in groups[1]["params"])
    assert backbone_count + head_count == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["lr"] == 1e-3


def test_adamw_optimizer_and_cosine_warmup_scheduler():
    model = FakeDINO()
    optimizer = create_optimizer(
        model,
        {
            "name": "adamw",
            "head_learning_rate": 1e-3,
            "backbone_learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
        },
    )
    scheduler = create_scheduler(
        optimizer,
        {
            "name": "cosine",
            "warmup_epochs": 1,
            "warmup_start_factor": 0.1,
            "min_learning_rate_ratio": 0.01,
        },
        total_epochs=4,
        steps_per_epoch=2,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert scheduler.get_last_lr() == pytest.approx([1e-6, 1e-4])


def test_learning_rate_multiplier_warms_up_then_decays():
    values = [
        learning_rate_multiplier(
            step=step,
            total_steps=10,
            warmup_steps=3,
            warmup_start_factor=0.1,
            scheduler_name="cosine",
            min_learning_rate_ratio=0.01,
        )
        for step in range(10)
    ]

    assert values[0] == 0.1
    assert values[2] == 1.0
    assert values[3] == 1.0
    assert values[-1] == 0.01


def test_run_epoch_updates_trainable_model():
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    images = torch.randn(6, 3, 2, 2)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        num_classes=3,
        optimizer=optimizer,
    )

    assert metrics["loss"] > 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


class DictionaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 3)

    def forward(self, inputs):
        return self.head(inputs["rgb"].mean(dim=(-2, -1)))


def test_run_epoch_accepts_dictionary_inputs():
    model = DictionaryModel()
    dataset = [
        ({"rgb": torch.randn(3, 2, 2)}, index % 3)
        for index in range(6)
    ]
    loader = DataLoader(dataset, batch_size=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        num_classes=3,
        optimizer=optimizer,
    )

    assert metrics["loss"] > 0.0
