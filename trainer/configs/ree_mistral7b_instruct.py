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
    per_device_train_batch_size=8,    # trzymamy bezpieczny batch na kartę
    gradient_accumulation_steps=16,   # efektywny batch 128
    learning_rate=2e-5,               # agresywniejszy LR
    warmup_ratio=0.03,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
)

