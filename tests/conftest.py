import pytest

from ftserve.finer import Example, Fact

TAGS = ["DebtInstrumentFaceAmount", "DebtInstrumentInterestRateStatedPercentage", "IncomeTaxExpenseBenefit"]

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture
def examples() -> list[Example]:
    return [
        Example(
            "t-1",
            "The Company issued $ 100 million of 7.00 % Senior Notes.",
            (Fact("100", TAGS[0]), Fact("7.00", TAGS[1])),
        ),
        Example("t-2", "Income tax expense was $ 12.5 million in 2019.", (Fact("12.5", TAGS[2]),)),
        Example("t-3", "The plan covers all full-time employees hired after January 1, 2015.", ()),
        Example("t-4", "The notes bear interest at 5.25 % per year.", (Fact("5.25", TAGS[1]),)),
    ]


@pytest.fixture(scope="session")
def tiny():
    """A 2-layer Qwen2 with a byte-level BPE tokenizer trained on the fly."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [
        "The Company issued $ 100 million of 7.00 % Senior Notes.",
        '{"facts": [{"value": "1", "tag": "x"}]}',
        "Income tax expense was $ 12.5 million in 2019. system user assistant",
    ] * 20
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=400,
            special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tokenizer.chat_template = CHAT_TEMPLATE
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    return tokenizer, cfg


def tiny_model(cfg):
    """Random weights, with embeddings scaled up: at the default init (std 0.02) the tied output
    layer cannot produce logits larger than about 0.6, so no adapter could fit anything."""
    import torch
    from transformers import Qwen2ForCausalLM

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        model.model.embed_tokens.weight.mul_(50)
    return model
