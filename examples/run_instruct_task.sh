"""
Config treningowy Ree – Mistral-7B-Instruct pod turniej.

To jest CZYSTY CONFIG (bez side-effectów).
Nic się samo nie odpala, dopóki:
- ktoś tego nie zaimportuje
- i nie zbuduje z tego TrainerProxyRequest / TrainingData itd.

Zaleta: wszystkie ważne decyzje (model, dataset, hyperparamy)
są w jednym miejscu, w czytelnej formie.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class InstructTaskConfig:
    """
    Ogólny config dla zadania typu InstructTextTask.

    Pola są dobrane tak, żeby:
    - dało się z nich łatwo zbudować payload dla trenera / walidatora,
    - były czytelne przy tuningu (zmieniasz liczby tutaj, a nie w 10 miejscach).
    """
    # Id taska – możesz zachować spójność z tym, co masz w bashu
    task_id: str

    # Typ zadania – w Twoim repo to jest InstructTextTask
    task_type: str

    # Model bazowy
    model_name_or_path: str

    # Dataset (HuggingFace name, S3 path, itd.)
    dataset_name: str
    dataset_split: str

    # Mapowanie pól datasetu na format oczekiwany przez InstructTextTask
    dataset_mapping: Dict[str, str]

    # Format:
    # - "hf"   → HuggingFace dataset
    # - "json" → lokalny json
    # - "csv"  → lokalny csv
    # - "s3"   → dane z S3
    file_format: str

    # Ile godzin walidator ma na skończenie taska
    hours_to_complete: int

    # Hyperparamy treningu
    max_steps: int
    max_seq_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    bf16: bool
    gradient_checkpointing: bool

    def to_training_payload(self) -> Dict[str, Any]:
        """
        Zwraca słownik, który można wrzucić w:
        - TrainingData / TrainerProxyRequest
        albo
        - bezpośrednio do jakiegoś launchera trenera.

        Przykład użycia (pseudo):

        payload = REE_MISTRAL7B_INSTRUCT.to_training_payload()
        TrainerProxyRequest(training_data=payload, hotkey=...)
        """
        return asdict(self)


# 🔥 Konkretny config „tryhard” dla 7B

REE_MISTRAL7B_INSTRUCT = InstructTaskConfig(
    task_id="ree-7b-instruct-tourn-001",
    task_type="InstructTextTask",

    # 7B model bazowy
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",

    # Instruct dataset – klasyczny Alpaca
    dataset_name="tatsu-lab/alpaca",
    dataset_split="train",

    # Mapowanie pól:
    # - instruction → prompt
    # - output      → target
    dataset_mapping={
        "field_instruction": "instruction",
        "field_output": "output",
    },

    file_format="hf",

    # Ile czasu walidator ma na trening
    hours_to_complete=16,

    # 🔥 Tryhard hyperparamy
    max_steps=3500,                   # więcej kroków niż 2500
    max_seq_length=2048,              # standard dla Mistrala
    per_device_train_batch_size=8,    # batch na jedną kartę
    gradient_accumulation_steps=16,   # efektywny batch 8*16 = 128
    learning_rate=2e-5,               # agresywniejszy LR
    warmup_ratio=0.03,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
)
