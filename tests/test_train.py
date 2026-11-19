import json

import torch
from conftest import TAGS, tiny_model

from ftserve.train import TrainConfig, answer_loss, collate, encode, lr_at, train


def test_encode_masks_the_prompt(tiny, examples):
    tokenizer, _ = tiny
    enc = encode(tokenizer, examples[0], TAGS)
    answer = [t for t, label in zip(enc["input_ids"], enc["labels"], strict=True) if label != -100]
    text = tokenizer.decode(answer)
    assert text.startswith('{"facts": [{"value": "100"') and text.rstrip().endswith("<|im_end|>")
    assert enc["labels"][0] == -100


def test_answer_loss_equals_full_masked_loss(tiny, examples):
    tokenizer, cfg = tiny
    model = tiny_model(cfg).eval()
    batch = collate([encode(tokenizer, e, TAGS) for e in examples[:2]], tokenizer.pad_token_id)
    ref = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss
    assert torch.allclose(answer_loss(model, batch), ref, atol=1e-5)


def test_lr_schedule():
    assert lr_at(0, 100, 10) == 0.1 and lr_at(9, 100, 10) == 1.0
    assert abs(lr_at(100, 100, 10)) < 1e-9


def test_train_lowers_loss_and_saves_adapter(tiny, examples, tmp_path):
    tokenizer, cfg = tiny
    model = tiny_model(cfg)
    c = TrainConfig(
        rank=4,
        alpha=8,
        dropout=0.0,
        lr=1e-2,
        epochs=15,
        batch_size=2,
        grad_accum=1,
        max_minutes=5,
        max_len=4096,
    )
    s = train(model, tokenizer, examples, examples, TAGS, c, tmp_path, log=lambda m: None)
    assert s["steps"] == 30 and not s["stopped_early"]
    assert s["val_loss_after"] < s["val_loss_before"] * 0.5
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    hist = json.loads((tmp_path / "train_history.json").read_text())
    assert len(hist["train"]) == 30
