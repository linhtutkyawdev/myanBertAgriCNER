# AGENT.md

# MyCNER — Burmese Agricultural Concept NER

## 1. Project Objective

Build and evaluate Burmese Agricultural Concept Named Entity Recognition (CNER)
models using two pretrained multilingual/base models:

1. Google mT5-small
2. UCSYNLP MyanBERTa

The primary objective is to determine how effectively each base model can be
fine-tuned for the existing Burmese agricultural CNER dataset.

The project must support:

- Reproducible dataset preparation
- Reproducible training
- Validation and test evaluation
- Model checkpointing
- Inference
- Entity-level evaluation
- Comparison between mT5 and MyanBERTa

The initial priority is **mT5-small**.

MyanBERTa will be implemented as the second model.

---

# 2. Existing Dataset

The source dataset is a text file where each sentence is represented as
pipe-separated token/label pairs.

Example:

    စပါး@CROP|စိုက်ပျိုး@FARM_OP|ရာတွင်@O|ဂျစ်ဆန်@FERT|နှင့်@O|ယူရီးယား@FERT

The format is:

    TOKEN@LABEL|TOKEN@LABEL|TOKEN@LABEL

The existing labels include:

    CROP
    FARM_OP
    FERT
    COUNT
    CROP_PART
    DIST
    NUT
    PERIOD
    DIS
    FUNG
    PEST
    SYM
    VAR
    TIME
    QTY
    SEASON
    PESTI
    EQUIP
    LOC
    ORG
    SOIL_TYPE
    PRICE
    WEED
    O

`O` means the token is not an entity.

Do not silently rename or remove existing labels.

The dataset format must be preserved as the source-of-truth format.

---

# 3. Project Principles

## 3.1 Keep the project simple

Do not introduce unnecessary frameworks.

Prefer:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets where useful
- Evaluate / seqeval where appropriate
- uv for environment and dependency management

Avoid unnecessary abstractions until the baseline works.

## 3.2 Reproducibility

All experiments must record:

- Base model
- Transformers version
- Tokenizers version
- Python version
- Dataset version
- Random seed
- Training configuration
- Maximum sequence length
- Batch size
- Learning rate
- Number of epochs
- Evaluation metrics

## 3.3 CPU-first development

The project must be runnable without a GPU.

Initial development should use:

- tiny dataset subset
- very small number of training steps
- small batch size
- short sequence length

The purpose of the first run is to verify the complete pipeline.

Do not optimize for training speed until correctness is established.

---

# 4. Clean Project Structure

Create the project from scratch with the following structure:

    mycner/
    │
    ├── AGENT.md
    ├── README.md
    ├── pyproject.toml
    ├── uv.lock
    │
    ├── data/
    │   ├── raw/
    │   │   └── sample_data.txt
    │   │
    │   ├── processed/
    │   │   ├── train.jsonl
    │   │   ├── validation.jsonl
    │   │   └── test.jsonl
    │   │
    │   └── labels.txt
    │
    ├── src/
    │   └── mycner/
    │       ├── __init__.py
    │       │
    │       ├── data/
    │       │   ├── __init__.py
    │       │   ├── parser.py
    │       │   ├── converter.py
    │       │   └── splitter.py
    │       │
    │       ├── models/
    │       │   ├── __init__.py
    │       │   ├── mt5.py
    │       │   └── myanberta.py
    │       │
    │       ├── training/
    │       │   ├── __init__.py
    │       │   ├── train_mt5.py
    │       │   └── train_myanberta.py
    │       │
    │       ├── evaluation/
    │       │   ├── __init__.py
    │       │   ├── evaluate_mt5.py
    │       │   ├── evaluate_myanberta.py
    │       │   └── metrics.py
    │       │
    │       └── inference/
    │           ├── __init__.py
    │           ├── mt5.py
    │           └── myanberta.py
    │
    ├── scripts/
    │   ├── prepare_data.py
    │   ├── inspect_dataset.py
    │   ├── train_mt5.py
    │   ├── train_myanberta.py
    │   └── evaluate.py
    │
    ├── experiments/
    │   ├── mt5/
    │   └── myanberta/
    │
    └── tests/
        ├── test_parser.py
        ├── test_converter.py
        └── test_alignment.py

The structure may be simplified if a component is unnecessary.

Do not create empty abstractions solely to satisfy this structure.

---

# 5. Environment Strategy

Use `uv`.

The project must not modify system Python.

Prefer a Python version compatible with the selected PyTorch and
Transformers versions.

Initially maintain one primary environment.

If MyanBERTa requires an incompatible Transformers/Tokenizers version,
maintain a separate environment rather than breaking the mT5 environment.

Example:

    .venv/

and, if necessary:

    .venv-myanberta/

The environments must be documented.

---

# 6. Phase 1 — Dataset Preparation

First implement the dataset parser.

Input:

    data/raw/sample_data.txt

Each non-empty line represents one annotated sentence.

Parse:

    TOKEN@LABEL|TOKEN@LABEL|...

into:

    {
        "tokens": [...],
        "labels": [...]
    }

Example:

    စပါး@CROP|စိုက်ပျိုး@FARM_OP|ရာတွင်@O

becomes:

    {
        "tokens": [
            "စပါး",
            "စိုက်ပျိုး",
            "ရာတွင်"
        ],
        "labels": [
            "CROP",
            "FARM_OP",
            "O"
        ]
    }

The parser must:

- Preserve Unicode
- Use UTF-8
- Preserve token ordering
- Preserve labels
- Reject malformed entries clearly
- Report malformed lines with line numbers

Do not silently discard malformed data.

---

# 7. Dataset Validation

Before training, generate a dataset report containing:

- Number of sentences
- Number of tokens
- Number of entities
- Number of labels
- Entity frequency by label
- Sentence length statistics
- Maximum sentence length
- Average sentence length
- Number of malformed records

Generate:

    data/labels.txt

containing the unique labels.

The label list must be deterministic.

`O` must always exist.

---

# 8. Train / Validation / Test Split

Split the dataset into:

- Train
- Validation
- Test

Recommended initial split:

    80% train
    10% validation
    10% test

Use a fixed random seed.

The same split must be reused for both mT5 and MyanBERTa so that their
results are directly comparable.

Do not randomly create a new split for each model.

Save the processed splits to:

    data/processed/train.jsonl
    data/processed/validation.jsonl
    data/processed/test.jsonl

---

# 9. mT5 Experiment

## 9.1 Base Model

Use:

    google/mt5-small

The initial objective is to fine-tune mT5-small as a
sequence-to-sequence CNER model.

---

# 10. mT5 Task Formulation

mT5 should be treated as a text-to-text generation model.

Do NOT initially force mT5 into standard token-classification BIO tagging.

Input:

    Burmese agricultural sentence

Target:

    The same sentence with entity labels inserted.

Example input:

    စပါး စိုက်ပျိုး ရာတွင် ဂျစ်ဆန် နှင့် ယူရီးယား ကို ၂ကြိမ် ခွဲ၍သုံးပါ၊၊

Target:

    စပါး<CROP> စိုက်ပျိုး<FARM_OP> ရာတွင် ဂျစ်ဆန်<FERT> နှင့် ယူရီးယား<FERT> ကို ၂ကြိမ်<COUNT> ခွဲ၍သုံးပါ၊၊

The exact serialization format must be deterministic.

---

# 11. mT5 Serialization Rules

Create a deterministic serializer.

For every source token:

If label is `O`:

    TOKEN

If label is an entity:

    TOKEN<LABEL>

Example:

    စပါး@CROP
    ရာတွင်@O
    ယူရီးယား@FERT

becomes:

    စပါး<CROP> ရာတွင် ယူရီးယား<FERT>

Do not add BIO prefixes to the mT5 target.

Do not alter the original Burmese text unnecessarily.

---

# 12. mT5 Special Tokens

The labels:

    <CROP>
    <FARM_OP>
    <FERT>
    ...

should be considered carefully as tokenizer vocabulary.

Before training:

- Inspect how mT5 tokenizes the label markers.
- Prefer adding entity marker tokens as additional special tokens if
  required for stable generation.
- Document the final decision.

The same target serialization must be used during:

- training
- validation
- inference
- evaluation

---

# 13. mT5 Dataset Format

The processed mT5 records should contain:

    {
        "input_text": "...",
        "target_text": "..."
    }

Example:

    {
        "input_text": "စပါး စိုက်ပျိုး ရာတွင် ...",
        "target_text": "စပါး<CROP> စိုက်ပျိုး<FARM_OP> ရာတွင် ..."
    }

Keep the original token/label representation available separately so that
evaluation can compare generated entities against the gold annotations.

---

# 14. mT5 Training

Use Hugging Face Transformers.

Preferred initial architecture:

    AutoTokenizer
    AutoModelForSeq2SeqLM

Base model:

    google/mt5-small

Initial CPU smoke test:

- Very small subset
- 1 epoch or very few steps
- Small batch size
- Short max sequence length

The first goal is:

    data → tokenizer → model → loss → checkpoint → inference

not model quality.

Once the pipeline works, train using the full training set.

---

# 15. mT5 Inference

Inference must accept:

    Burmese agricultural text

and generate:

    labeled CNER text

Example:

Input:

    မြေဆီလွှာတွင် နိုက်ထရိုဂျင် ချို့တဲ့မှု ဖြစ်ပေါ်နိုင်သည်။

Expected structure:

    မြေဆီလွှာ<SOIL_TYPE> တွင် နိုက်ထရိုဂျင်<NUT> ...

The inference code must parse the generated representation back into:

    [
        {
            "text": "...",
            "label": "..."
        }
    ]

Do not rely solely on visual inspection of generated text.

---

# 16. mT5 Evaluation

Evaluation must eventually report:

- Entity precision
- Entity recall
- Entity F1
- Per-label precision
- Per-label recall
- Per-label F1
- Exact entity match count
- Invalid/malformed generated output count

Evaluation should operate on structured extracted entities, not raw generated
strings alone.

---

# 17. MyanBERTa Experiment

After mT5 is working, implement:

    UCSYNLP/MyanBERTa

MyanBERTa should use the standard encoder/token-classification architecture.

Preferred architecture:

    AutoTokenizer
    AutoModelForTokenClassification

with:

    num_labels = number of CNER labels

The labels should be converted to a token-classification representation.

---

# 18. MyanBERTa Label Strategy

For MyanBERTa, use BIO tagging.

Example source:

    စပါး@CROP
    စိုက်ပျိုး@FARM_OP
    ရာတွင်@O

becomes:

    စပါး       B-CROP
    စိုက်ပျိုး   B-FARM_OP
    ရာတွင်     O

If an entity spans multiple dataset tokens:

    B-LABEL
    I-LABEL
    I-LABEL
    ...

The conversion must preserve the original entity boundaries.

---

# 19. MyanBERTa Subword Alignment

MyanBERTa uses subword tokenization.

A single dataset token may produce multiple model tokens.

The implementation must correctly align the original BIO label to the
resulting subword tokens.

Preferred approach:

- Use tokenizer offset mappings when supported.
- Otherwise use word IDs / tokenizer-provided alignment.
- Special tokens receive label `-100`.
- Labels ignored with `-100` must not contribute to loss.

Document the exact alignment strategy.

---

# 20. MyanBERTa Training

Use:

    AutoModelForTokenClassification

Train using the same:

- train split
- validation split
- test split
- random seed

used for mT5.

This is necessary for a fair comparison.

---

# 21. MyanBERTa Evaluation

Report:

- Token-level precision
- Token-level recall
- Token-level F1
- Entity-level precision
- Entity-level recall
- Entity-level F1
- Per-label metrics

The primary comparison metric should be entity-level F1.

---

# 22. Final Model Comparison

After both models work, produce a comparison report.

Example:

    Model          Precision    Recall    F1
    ------------------------------------------------
    mT5-small       ...          ...      ...
    MyanBERTa       ...          ...      ...

Also compare:

- Training time
- Inference time
- Model size
- CPU memory usage
- GPU memory usage if available
- Number of parameters
- Sequence/tokenization behavior
- Per-label performance
- Failure cases

Do not declare a winner based only on overall F1.

Analyze which entity categories benefit from each model.

---

# 23. Experiment Tracking

Each experiment should save:

    experiments/
        mt5/
            config.json
            metrics.json
            predictions.jsonl
            checkpoints/

        myanberta/
            config.json
            metrics.json
            predictions.jsonl
            checkpoints/

Each experiment must record:

- model name
- model revision if relevant
- dataset version/hash
- random seed
- hyperparameters
- environment versions
- training duration
- evaluation metrics

---

# 24. Inference API / CLI

Eventually provide a simple command such as:

    uv run python scripts/predict.py \
        --model mt5 \
        --text "မြေဆီလွှာတွင် နိုက်ထရိုဂျင် ချို့တဲ့မှု ဖြစ်ပေါ်နိုင်သည်။"

and:

    uv run python scripts/predict.py \
        --model myanberta \
        --text "မြေဆီလွှာတွင် နိုက်ထရိုဂျင် ချို့တဲ့မှု ဖြစ်ပေါ်နိုင်သည်။"

Both should return structured entities.

---

# 25. Testing

Tests must cover:

## Dataset parser

- Normal record
- Multiple entities
- `O` labels
- Unicode Burmese
- Malformed records
- Empty lines

## mT5 serialization

- `O` entity
- Single entity
- Multiple entities
- Multiple labels
- Burmese Unicode

## MyanBERTa alignment

- Single-token entity
- Multi-token entity
- Subword split
- Special tokens
- Sentence boundaries

## Evaluation

- Perfect prediction
- Missing entity
- Spurious entity
- Wrong label
- Partial entity match

---

# 26. Development Order

Do NOT implement the entire project at once.

Follow this order.

## Phase A — Clean project

1. Create repository
2. Create uv environment
3. Add dependencies
4. Create project structure
5. Add raw dataset
6. Add basic README

## Phase B — Dataset

1. Implement parser
2. Validate parser
3. Generate dataset statistics
4. Extract label list
5. Create deterministic train/validation/test split
6. Save JSONL datasets

## Phase C — mT5

1. Load tokenizer
2. Verify Burmese text
3. Verify label-marker tokenization
4. Create input/target pairs
5. Run dataset preprocessing
6. Run CPU smoke test
7. Run inference
8. Implement evaluation
9. Train full model
10. Save checkpoint

## Phase D — MyanBERTa

1. Load compatible tokenizer
2. Verify Burmese text
3. Convert labels to BIO
4. Implement subword alignment
5. Run alignment tests
6. Run CPU smoke test
7. Run inference
8. Implement evaluation
9. Train full model
10. Save checkpoint

## Phase E — Comparison

1. Evaluate both on identical test data
2. Generate metrics
3. Compare per-label results
4. Compare inference performance
5. Inspect failure cases
6. Document conclusions

---

# 27. Current Priority

The immediate task is ONLY:

    Clean project setup
        ↓
    Dataset parser
        ↓
    Dataset conversion
        ↓
    mT5 train/validation/test JSONL

Do not start model training until the processed dataset has been inspected.

The first milestone is:

    data/raw/sample_data.txt
             ↓
             parser
             ↓
    data/processed/train.jsonl
    data/processed/validation.jsonl
    data/processed/test.jsonl

Each record for mT5 must contain:

    input_text
    target_text

and the conversion must be manually inspected before proceeding.

---

# 28. Current Model Priority

Priority order:

    1. mT5-small
    2. MyanBERTa

mT5 is the first complete experimental pipeline.

MyanBERTa is the second experimental pipeline.

Both must ultimately use the same underlying dataset split.

---

# 29. Important Constraints

- Do not modify the original raw dataset.
- Do not silently remove records.
- Do not silently change entity labels.
- Do not change Burmese Unicode normalization without documenting it.
- Do not use a different train/test split for different models.
- Do not compare models using different test sets.
- Do not judge tokenizer quality solely by token count.
- Do not judge model quality from training loss alone.
- Do not claim a model works based solely on successful loading.
- Always evaluate on unseen test data.
- Keep CPU smoke tests separate from full experiments.

---

# 30. Definition of Done

The project is considered functional when:

1. The raw CNER dataset is parsed correctly.
2. Train/validation/test datasets are reproducible.
3. mT5-small can be fine-tuned on the dataset.
4. mT5-small can generate CNER predictions.
5. Generated predictions can be parsed into structured entities.
6. mT5 receives entity-level evaluation metrics.
7. MyanBERTa can be fine-tuned using aligned BIO labels.
8. MyanBERTa receives entity-level evaluation metrics.
9. Both models are evaluated on the same test set.
10. Results are saved and reproducible.
11. A final comparison can be produced from the recorded experiments.

The immediate implementation target is **Phase A → Phase B → mT5 dataset preparation**.