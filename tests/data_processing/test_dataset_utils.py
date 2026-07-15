import os
from types import SimpleNamespace

from scaletraining.data_processing.dataset_utils import (
    get_dataset_text_files,
    load_hf_dataset,
    local_text_dataset_files,
)
from scaletraining.data_processing.tokenizer import TextTokenizer


def test_local_text_dataset_directory_loads_train_and_validation(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    train = corpus / "train.txt"
    validation = corpus / "validation.txt"
    train.write_text("one training line\nanother training line\n", encoding="utf-8")
    validation.write_text("one validation line\n", encoding="utf-8")

    files = local_text_dataset_files(str(corpus))
    dataset = load_hf_dataset([str(corpus)], [None])

    assert files["train"] == train
    assert files["validation"] == validation
    assert set(dataset.keys()) == {"train", "validation"}
    assert dataset["train"][0]["text"] == "one training line"


def test_get_dataset_text_files_reuses_local_text_files(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    train = corpus / "train.txt"
    validation = corpus / "validation.txt"
    train.write_text("train\n", encoding="utf-8")
    validation.write_text("validation\n", encoding="utf-8")
    cfg = SimpleNamespace(
        tokenizer=SimpleNamespace(dataset_names=[str(corpus)], dataset_tag=[None]),
        paths=SimpleNamespace(tokenizer_train_data=str(tmp_path / "raw")),
    )

    assert get_dataset_text_files(cfg) == [str(train), str(validation)]


def test_custom_tokenizer_trained_from_local_text_has_eos_id(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train.txt").write_text("alpha beta gamma\n", encoding="utf-8")
    cfg = SimpleNamespace(
        tokenizer=SimpleNamespace(
            dataset_names=[str(corpus)],
            dataset_tag=[None],
            is_pretrained=False,
            custom_tokenizer_vocab_size=64,
        ),
        paths=SimpleNamespace(tokenizer_train_data=str(tmp_path / "raw")),
    )

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        tok = TextTokenizer(cfg)
    finally:
        os.chdir(old_cwd)

    assert tok.eos_id is not None
    assert isinstance(tok.eos_id, int)
