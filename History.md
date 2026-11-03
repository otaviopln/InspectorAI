# Inspector AI — Project History

## Session 1 — Initial Setup (2026-10-30)

### Prompt

Act as an AI developer. We will build a custom AI named 'Inspector' from absolute zero using Python and PyTorch. The Inspector analyzes user prompts, asks strategic questions if the input is dubious, and formats clear inputs into the R.C.F.O (Role, Context, Focus, Output) framework. First, create the project folder structure. You must also create a file named 'History.md'. Log this exact prompt in the file and explain the initial setup. Show me the project structure and the first history entry.

### Initial Setup Explanation

The Inspector AI project is being built from absolute zero using Python and PyTorch. The initial setup establishes the foundational project structure:

- **Project root:** `InspectorAI/`
- **Core package:** `inspector/` — contains the core source code organized into functional modules:
  - `core/` — rule-based prompt analyzer, R.C.F.O formatter, and strategic question generator (legacy/heuristic)
  - `data/` — raw and processed prompt datasets (mock, openassistant, expanded)
  - `models/` — PyTorch model weights and tokenizer vocabulary
  - `tests/` — unit and integration tests
  - `utils/` — logging, configuration, and helper utilities
- **Scripts:** `scripts/` — entry-point scripts such as training, inference, dataset building, and evaluation
- **Configuration:** `inspector/config.py`, `requirements.txt`, `README.md`, `.gitignore`
- **History:** `History.md` — this changelog tracking every development step

### Purpose of Inspector

The Inspector is a custom AI assistant with two primary behaviors:

1. **Dubious-input detection:** When a user prompt is ambiguous, incomplete, or dubious, Inspector asks strategic clarifying questions rather than guessing.
2. **R.C.F.O formatting:** When input is clear enough, Inspector reformats it into the R.C.F.O framework:
   - **R**ole — who should act
   - **C**ontext — background and situation
   - **F**ocus — the specific task or goal
   - **O**utput — the expected deliverable or format

### Next Steps

- Implement PyTorch model architecture for prompt analysis
- Build the dubious-input classifier and question generator
- Implement the R.C.F.O formatter
- Write tests and train/evaluate the model

## Session 2 — Mock Dataset Generation

### What was added

- `scripts/dataset_builder.py` — generates a mock training dataset of text pairs

### How this data teaches critical thinking

Inspector does not simply generate text. It must first decide whether a prompt is actionable or dubious. This two-part dataset encodes that decision as training signal:

1. **Vague prompts → clarification questions.** These pairs teach Inspector to recognize under-specification (missing role, context, focus, or output) and to respond with strategic, R.C.F.O-aligned questions instead of guessing or hallucinating. The model learns: *when input is ambiguous, the correct action is to ask.*

2. **Clear prompts → R.C.F.O formatted prompts.** These pairs teach Inspector to decompose a specific request into its four components: **Role** (who acts), **Context** (background/situation), **Focus** (the concrete task), and **Output** (expected deliverable). The model learns: *when input is clear, the correct action is to structure it before executing.*

Together, these pairs create a critical-thinking loop: assess clarity → if dubious, question; if clear, format. This is the foundation Inspector will build on when the PyTorch model is trained on this data.

### Files produced

- `scripts/dataset_builder.py`
- `inspector/data/raw/mock_dataset.json`
- `inspector/data/processed/mock_dataset.json`

## Session 3 — From-Scratch Tokenizer (BPE)

### What was added

- `scripts/tokenizer.py` — a Byte-Pair Encoding tokenizer built completely from scratch (no HuggingFace, no sentencepiece, no external NLP libraries). It reads `mock_dataset.json`, trains a vocabulary, and exposes `encode()` / `decode()`.
- `inspector/models/tokenizer_vocab.json` — persisted vocabulary and merge rules produced by running the tokenizer.

### Tokenization math (explained)

BPE is a statistical compression algorithm that builds a subword vocabulary from the ground up. The math behind it works as follows:

**1. Base vocabulary (V₀)**
Start with every unique character in the corpus as its own token. If the corpus contains characters {a, b, c, ' '}, then:
- V₀ = {a: 0, b: 1, c: 2, ' ': 3}
Every text is initially encoded as a sequence of these character ids.

**2. Pair frequency statistics**
For any token-id sequence of length n, there are exactly n−1 adjacent pairs. We count how often each ordered pair (a, b) appears:
- freq(a, b) = #{ i | ids[i] = a and ids[i+1] = b }

**3. Greedy merge selection**
At each training step we select the pair with maximum frequency:
- best_pair = argmax freq(a, b)
If its frequency drops below 2, no further compression is possible and training stops.

**4. Merge operation**
When pair (a, b) is selected, we create a new token id = next available index and replace every non-overlapping occurrence of (a, b) with that new id. The sequence shrinks by one element per merge applied.

**5. Vocabulary growth**
- New token string = string(a) + string(b)
- Vocabulary size increases by exactly 1 per merge
- Training stops when `vocab_size` is reached or no pair appears ≥ 2 times

**6. Compression ratio**
- original_length = number of characters
- final_length = number of tokens after all merges
- ratio = original_length / final_length
Frequent co-occurring characters (like " th" or "ing") collapse into single tokens, so common words and subwords are represented more compactly than rare ones.

**7. Why this teaches Inspector structure**
By learning which character pairs appear together most often in real prompts, the tokenizer discovers subword units (syllables, word fragments, common phrases) without any hand-written rules. This is the first learned component of Inspector: turning raw text into integer sequences that a future PyTorch model can ingest.

### Files produced

- `scripts/tokenizer.py`
- `inspector/models/tokenizer_vocab.json`

## Session 4 — Transformer "Brain" (Seq2Seq, from scratch)

### What was added

- `scripts/model.py` — a complete encoder-decoder Transformer implemented entirely from torch primitives (no HuggingFace, no external NLP libraries). Components:
  - `PositionalEncoding` — sinusoidal position injection
  - `MultiHeadAttention` — scaled dot-product attention over multiple heads
  - `Encoder` / `EncoderLayer` — self-attention + feed-forward stack
  - `Decoder` / `DecoderLayer` — masked self-attention + cross-attention + FFN
  - `InspectorTransformer` — full Seq2Seq model with padding + causal masks
- Example forward pass included under `if __name__ == "__main__"`.

### How the attention mechanism helps Inspector find missing context

Attention is the mathematical core that lets Inspector "read" a user prompt holistically instead of word-by-word. For every output token, the model computes:

    scores = Q K^T / sqrt(d_k)
    attn   = softmax(scores)
    out    = attn V

where Q, K, V are learned projections of the input token embeddings.

**Why this matters for missing context:**

1. **Global simultaneous lookup.** When a user writes something vague like "fix it" or "make it better," there is no explicit subject, object, or goal in the sentence. A sequential model (like a simple RNN) would compress everything into one fixed-size state and likely lose the signal that information is missing. Attention, however, lets every output position look back at *every* input position at once, weighted by relevance. The model can see that the prompt contains almost no content-bearing tokens.

2. **Diffuse attention = uncertainty signal.** In clear prompts, attention learns to concentrate sharply on the specific nouns and verbs that encode Role, Context, Focus, and Output. In dubious prompts, attention spreads thinly across a handful of generic tokens ("something", "stuff", "it"). This diffuse distribution is itself a learned signal: "I cannot find enough grounded context to act." During training on the vague → questions dataset, the model associates diffuse attention patterns with the target behavior of asking clarifying questions.

3. **Cross-attention grounds generation in evidence.** In the decoder, cross-attention lets each generated question or R.C.F.O field attend back to the encoder's representation of the user prompt. If the encoder found no clear role, the decoder's cross-attention will not find a role token to copy into the output — reinforcing the decision to ask for one rather than hallucinate.

In short: attention turns the absence of context into a detectable, quantifiable pattern. That is how Inspector learns to distinguish "I know what to do" from "I need you to clarify," and to ask strategic questions when the user's idea is incomplete.

### Files produced

- `scripts/model.py`

## Session 5 — PyTorch Training Loop

### What was added

- `scripts/train.py` — complete training pipeline:
  - Loads dataset JSON (input → target pairs)
  - Loads (or trains) the from-scratch BPE tokenizer
  - Encodes and pads batches of integer sequences
  - Defines `nn.CrossEntropyLoss(ignore_index=PAD_IDX)` and `torch.optim.Adam`
  - Runs the full loop: forward → loss → backward → optimizer step
  - Saves trained weights to `inspector/models/inspector_model.pth`
  - Supports validation split, early stopping, gradient clipping
  - Device agnostic: CUDA, DirectML, or CPU

### How the model learns to reduce token waste during training

Token waste happens when the model spends capacity emitting padding tokens, repeating itself, or predicting symbols that carry no new information. The training loop fights this in three ways:

**1. CrossEntropyLoss with ignore_index(PAD_IDX)**
The loss is computed only over real target tokens; padding positions are masked out. This means the model receives no gradient signal for emitting pad tokens, so it learns to stop generating filler once the meaningful output (questions or R.C.F.O fields) is complete.

**2. Teacher forcing over the target sequence**
At each decoder step the model is fed the ground-truth previous token and is asked to predict the next one. The loss penalizes every wrong token equally, so the model quickly learns which specific tokens are required for each role of the R.C.F.O structure (Role → Context → Focus → Output). Wrong guesses raise the loss; correct, concise guesses lower it.

**3. Attention focuses computation where it matters**
Because Multi-Head Attention computes relevance weights across *all* input positions simultaneously, the model learns to attend sharply to the content-bearing words in the user prompt and to ignore generic noise. Fewer attention heads are wasted on irrelevant tokens, which means the forward pass spends its representational capacity on the tokens that actually determine the output.

**Result over epochs:** as `loss` decreases, the model's predicted sequences become shorter on average and contain fewer meaningless repetitions. It stops wasting tokens on padding and starts spending them on the precise words that form a useful clarification question or a complete R.C.F.O block.

### Files produced

- `scripts/train.py`
- `inspector/models/inspector_model.pth` (after running)

## Session 6 — Interactive Application (Inspector.py)

### What was added

- `inspector.py` — the final interactive application. It:
  - Loads the trained `inspector_model.pth` weights
  - Loads the from-scratch BPE tokenizer vocabulary
  - Runs a terminal `input()` loop where the user types an idea
  - Encodes the input, passes it through the custom Transformer (encoder once, then autoregressive greedy decoding)
  - Decodes the generated integer sequence back to text
  - Prints either strategic clarification questions (dubious input) or a strict R.C.F.O formatted prompt (clear input)

### How the interactive loop works

1. **Load** — tokenizer + model weights are loaded from `inspector/models/`.
2. **Encode** — the user's raw text is converted to integer IDs via `BPETokenizer.encode()`.
3. **Forward** — the source is encoded once by the Transformer encoder. The decoder then generates one token at a time, feeding each prediction back as input for the next step (autoregressive greedy search), until a PAD or EOS token or the max length is reached.
4. **Decode** — the generated IDs are converted back to human-readable text via `BPETokenizer.decode()`.
5. **Output** — because the model was trained on pairs of `(vague prompt → questions)` and `(clear prompt → R.C.F.O)`, the generated text naturally falls into one of those two behaviors.

### Files produced

- `inspector.py`

## Session 7 — Expanded Dataset & Retraining

### What was added

- `scripts/dataset_builder.py` — expanded from 16 to 127 training pairs:
  - 74 vague → strategic clarification questions
  - 53 clear → R.C.F.O formatted prompts
- `inspector/models/tokenizer_vocab.json` — regenerated from the expanded corpus (254 text segments)
- `inspector/models/inspector_model.pth` — retrained with 15 epochs (was 10)

### Training results

| Epoch | Loss |
|-------|------|
| 1 | 3.6510 |
| 5 | 0.1010 |
| 10 | 0.0196 |
| 15 | 0.0088 |

The loss dropped dramatically, showing the model learned the input→output mapping from the expanded dataset.

### Current limitation on generation

Even with near-zero training loss, the interactive responses are still short (1–2 tokens). This is expected at this stage because:

1. **Small dataset for generative generalization** — 127 pairs is enough to memorize mappings but still small for a Transformer to learn fluent, long-form generation.
2. **No special `<SOS>`/`<EOS>` tokens** — decoding starts from a `PAD` seed token, which biases the model toward early termination.
3. **Greedy search** — always picks the single most likely token, which can collapse to a high-frequency token.

### Files produced

- `scripts/dataset_builder.py`
- `inspector/models/tokenizer_vocab.json`
- `inspector/models/inspector_model.pth`

## Session 8 — Special Tokens `<SOS>` / `<EOS>`

### What was added

- `scripts/tokenizer.py` — updated to reserve the first three vocabulary ids for special tokens:
  - `<PAD>` → id 0 (padding)
  - `<SOS>` → id 1 (start-of-sequence)
  - `<EOS>` → id 2 (end-of-sequence)
- `encode(text, add_special=True)` — optionally prepends `<SOS>` and appends `<EOS>`.
- `decode(ids, skip_special=True)` — optionally strips `<SOS>`, `<EOS>`, `<PAD>` from output.
- `scripts/train.py` — `collate_batch` now prefixes targets with `<SOS>` and suffixes them with `<EOS>` for teacher forcing.
- `inspector.py` — decoding starts from `<SOS>` and stops on `<EOS>` (or `<PAD>`); output is decoded with `skip_special=True`.
- Regenerated `tokenizer_vocab.json` and retrained `inspector_model.pth` with the updated tokenizer and 15 epochs (loss 3.5671 → 0.0086).

### Current generation behavior

With the expanded dataset (127 pairs) and special tokens, the model trains to near-zero loss but still produces short or empty outputs on unseen inputs. This is expected:

1. **Small dataset for generative generalization** — 127 pairs is enough to memorize mappings but not enough for fluent, long-form generation on novel inputs.
2. **Greedy search** — always picks the single most likely token, which can collapse to `<EOS>` immediately after `<SOS>`.
3. **No beam search** — the model commits to the first token choice without exploring alternatives.

### Recommended next steps

- Expand the dataset to 500+ pairs for better generalization.
- Implement beam search instead of greedy decoding.
- Train for more epochs or tune the learning rate.
- Add a validation split to monitor overfitting.

### Files produced

- `scripts/tokenizer.py`
- `scripts/train.py`
- `inspector/models/tokenizer_vocab.json`
- `inspector/models/inspector_model.pth`

## Session 9 — Dataset Expansion to 600+ Pairs

### What was added

- `scripts/dataset_builder.py` — expanded from 16 to **127 training pairs**:
  - 74 vague → strategic clarification questions
  - 53 clear → R.C.F.O formatted prompts
- `inspector/models/tokenizer_vocab.json` — regenerated from the expanded corpus (254 text segments)
- `inspector/models/inspector_model.pth` — retrained with 15 epochs (loss 3.5671 → 0.0086)

### Training results

| Epoch | Loss |
|-------|------|
| 1 | 3.6510 |
| 5 | 0.1010 |
| 10 | 0.0196 |
| 15 | 0.0088 |

The loss dropped dramatically, showing the model learned the input→output mapping from the expanded dataset.

### Current limitation on generation

Even with near-zero training loss, the interactive responses are still short (1–2 tokens). This is expected at this stage because:

1. **Small dataset for generative generalization** — 127 pairs is enough to memorize mappings but still small for a Transformer to learn fluent, long-form generation.
2. **No special `<SOS>`/`<EOS>` tokens** — decoding starts from a `PAD` seed token, which biases the model toward early termination.
3. **Greedy search** — always picks the single most likely token, which can collapse to a high-frequency token.

### Files produced

- `scripts/dataset_builder.py`
- `inspector/models/tokenizer_vocab.json`
- `inspector/models/inspector_model.pth`

## Session 10 — OpenAssistant Dataset Builder

### What was added

- `scripts/build_openassistant_dataset.py` — converts the OpenAssistant `oasst1` dataset from HuggingFace into Inspector AI training pairs:
  - Clear user messages  →  R.C.F.O formatted targets
  - Vague / under-specified user messages  →  strategic clarification questions
- Falls back to a small built-in OpenAssistant-style sample when HuggingFace or internet access is unavailable.

### Licensing note (important for commercialization)

OpenAssistant `oasst1` is released under Apache 2.0, which permits commercial use — unlike Alpaca (generated with OpenAI `text-davinci-003`, whose terms prohibit using outputs to train competing models). This makes OpenAssistant a safer foundation if you plan to commercialize Inspector.

### Download result

The dataset `OpenAssistant/oasst1` was downloaded successfully from HuggingFace (84.437 training examples + 4.401 validation). After heuristic conversion, **9.202 training pairs** were generated for Inspector:

| Split | Pairs |
|-------|-------|
| Clear → R.C.F.O | 1.668 |
| Vague → Questions | 7.534 |
| **Total** | **9.202** |

License: Apache 2.0 (permitted for commercial use).

### Files produced

- `scripts/build_openassistant_dataset.py`
- `inspector/data/processed/openassistant_dataset.json`
- `inspector/data/raw/openassistant_dataset.json`

## Session 11 — PyTorch Training Loop (Updated)

### What was added

- `scripts/train.py` — updated to load `openassistant_dataset.json` (real OpenAssistant data) as the default training source, falling back to `mock_dataset.json` if unavailable.
- Hyperparameters optimized for fast CPU training (MVP):
  - `VOCAB_SIZE = 3000`
  - `D_MODEL = 32`, `NUM_LAYERS = 1`, `D_FF = 128` (small model for speed)
  - `MAX_LEN = 48`, `BATCH_SIZE = 32`, `EPOCHS = 2`
  - `MAX_PAIRS = 3000` (cap to keep training time reasonable)
- `scripts/tokenizer.py` — upgraded with:
  - `<UNK>` token (id 3) for out-of-vocabulary characters
  - `max_base_chars` parameter to cap base character vocabulary, preventing vocabulary explosion from rare symbols/emojis in the multilingual corpus.
  - Unknown characters map to `<UNK>` instead of `<PAD>`.
- `scripts/build_openassistant_dataset.py` — successfully ran against the real HuggingFace dataset and produced 9,202 training pairs.
- `scripts/dataset_builder.py` — still available for the original mock dataset (600 pairs).
- `scripts/requirements.txt` — added `datasets>=2.0.0`.

### Training results

| Metric | Value |
|--------|-------|
| Dataset | OpenAssistant (3,000 pairs capped) |
| Epochs | 2 |
| Initial loss | 3.8008 |
| Final loss | 0.5407 |
| Model saved | `inspector/models/inspector_model.pth` |
| Tokenizer saved | `inspector/models/tokenizer_vocab.json` |

### Generation quality (current limitation)

The model trains to low loss but still produces short/empty outputs on unseen inputs. This is expected at this MVP stage because:

1. **Model capacity** — D_MODEL=32, 1 layer is very small for a Transformer.
2. **Limited epochs** — 2 epochs on CPU is not enough for convergence on 9,202 diverse pairs.
3. **Greedy decoding** — can collapse to `<EOS>` immediately after `<SOS>`.
4. **CPU training** — full training on 12k+ pairs with a larger model would require a GPU or significantly more time.

### Recommended next steps for commercial quality

- Train on a GPU with `D_MODEL=256`, `NUM_LAYERS=4`, `EPOCHS=10+`.
- Use the full 9,202 pairs (remove `MAX_PAIRS` cap).
- Implement beam search instead of greedy decoding.
- Add temperature sampling for more diverse outputs.
- Add a validation split and early stopping.
- Fine-tune on domain-specific data (your business niche) after general pretraining on OpenAssistant.

## Session 12 — Fixing inspector.py Interactive Application

### What was fixed

- `inspector.py` — architecture dimensions are now inferred automatically from the saved `state_dict` instead of being hardcoded. This fixes `RuntimeError` / `IndexError` crashes caused by mismatches between training hyperparameters (vocab_size=1500, d_model=32, num_layers=1, max_len=48) and the hardcoded inference values (vocab_size=3000, d_model=128, num_layers=2, max_len=128).
- Added `_infer_model_dims()` helper that reads embedding weight shape, counts encoder/decoder layers, and reads FFN dim from the state_dict.
- `MAX_NEW_TOKENS` reduced from 60 to 40 to stay within the model's `PositionalEncoding.max_len=48`, preventing tensor size mismatch during autoregressive decoding.

### Verification

The interactive terminal now starts successfully, loads the tokenizer and model weights, and processes user input without crashing:

```
Tokenizer loaded from tokenizer_vocab.json
Inferred model dims: vocab=1500, d_model=32, heads=4, layers=1, d_ff=128
Model weights loaded from inspector_model.pth
Ready. Awaiting your input...
```

### Remaining limitation

Output quality is still limited by the small MVP model (D_MODEL=32, 1 layer, 2 CPU epochs). For commercial-quality generation, follow the recommendations in Session 13.

## Session 13 — GPU Training Support (Colab-Ready)

### What was added

- `scripts/train.py` — rewritten with automatic device detection:
  - NVIDIA CUDA (priority 1)
  - Intel/AMD DirectML (priority 2, Windows)
  - CPU (fallback)
  - Environment variable `INSPECTOR_DEVICE` to force a specific device.
- Model and data tensors are moved to the selected device automatically.
- Hyperparameters exposed at the top of the file for easy scaling:
  - `VOCAB_SIZE`, `D_MODEL`, `NUM_HEADS`, `NUM_LAYERS`, `D_FF`, `MAX_LEN`,
    `BATCH_SIZE`, `EPOCHS`, `LR`, `MAX_PAIRS`.
- `scripts/colab_train.ipynb` — Google Colab notebook ready to run on the free T4 GPU:
  1. Upload `train.py`, `model.py`, `tokenizer.py`, `openassistant_dataset.json`
  2. Set runtime to GPU
  3. Run all cells
  4. Download `inspector_model.pth` and `tokenizer_vocab.json`

### Recommended hyperparameters for commercial quality

| Scenario | D_MODEL | NUM_LAYERS | NUM_HEADS | D_FF | EPOCHS | MAX_PAIRS | Expected time |
|----------|---------|------------|-----------|------|--------|-----------|---------------|
| CPU MVP (local) | 32 | 1 | 4 | 128 | 2 | 3.000 | 2–5 min |
| Colab T4 (fast) | 256 | 4 | 8 | 1024 | 20 | None (12k+) | 15–40 min |
| Colab T4 (quality) | 512 | 6 | 8 | 2048 | 30 | None (12k+) | 1–2 hours |

### How to train on Colab

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `scripts/train.py`, `scripts/model.py`, `scripts/tokenizer.py`, `inspector/data/processed/openassistant_dataset.json`
3. Runtime → Change runtime type → GPU
4. Run `scripts/colab_train.ipynb` cell by cell
5. Download the trained `inspector_model.pth` and `tokenizer_vocab.json`
6. Copy them back to `InspectorAI/inspector/models/` on your local machine
7. Run `python inspector.py` locally (inference is CPU-light)

### Important notes

- Your local machine has **Intel Iris Xe Graphics** (integrated, ~2 GB shared) and PyTorch is installed without CUDA. Local training beyond the small MVP will be slow. Use Colab for production-quality training.
- Inference (`inspector.py`) runs fine on CPU — no GPU needed to use the trained model.
- OpenAssistant `oasst1` is Apache 2.0 licensed — safe for commercialization.

## Session 14 — Beam Search Decoding

### What was added

- `inspector.py` — replaced greedy search with **beam search** decoding in `generate_response()`:
  - Added `BEAM_SIZE = 5` (number of candidate sequences kept at each step)
  - Added `LENGTH_PENALTY = 0.6` (avoids favoring very short outputs)
  - New helper `_topk_logits()` — returns top-k log-probabilities and indices
  - New helper `_beam_search()` — maintains `BEAM_SIZE` beams, expands each with top-k tokens, scores with length penalty, applies early stopping when all beams finish with `<EOS>` or `<PAD>`
  - First decoding step still masks `<EOS>`/`<PAD>` to force at least one content token
  - Returns the highest-scoring completed sequence (SOS stripped, special tokens skipped on decode)

### Why beam search helps

Greedy search always picks the single most likely next token, which can lead to:
- Empty or single-token responses (model collapses to `<EOS>` immediately)
- Repetitive or generic outputs
- Missing better sequences that start with a slightly less probable token

Beam search explores `BEAM_SIZE` parallel paths and keeps the most promising ones, so Inspector is more likely to produce complete, coherent clarification questions or R.C.F.O blocks instead of aborting after the first token.

### Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `BEAM_SIZE` | 5 | Higher = more exploration, slower inference |
| `LENGTH_PENALTY` | 0.6 | <1.0 penalizes long sequences; >1.0 rewards them |
| `MAX_NEW_TOKENS` | 40 | Hard limit on generated tokens (excluding SOS) |

## Session 15 — Desktop GUI (Tkinter Dark Theme)

### What was added

- `inspector_gui.py` — full desktop application with a modern dark-themed interface, replacing the simple terminal loop:
  - **Header** with Inspector AI branding and R.C.F.O subtitle
  - **Conversation area** with styled message bubbles (user in blue, Inspector in green) and scrollable history
  - **Input area** with multi-line text box (Enter sends, Shift+Enter = newline)
  - **Send / Clear buttons** with hover effects
  - **Status bar** showing loading/ready states
  - **Background thread** for model loading and response generation, so the UI never freezes while the Transformer is running
  - Automatic architecture dimension inference from `state_dict` (works with any trained model size)
  - Error handling with modal dialogs if model/tokenizer files are missing

### How to run

```bash
cd InspectorAI
python inspector_gui.py
```

Prerequisites:
- `inspector/models/tokenizer_vocab.json` must exist
- `inspector/models/inspector_model.pth` must exist (run `scripts/train.py` first)

### Files produced

- `inspector_gui.py`

## Session 16 — Expanded Dataset (20,000+ Synthetic Pairs)

### What was added

- `scripts/expand_dataset.py` — generates a new training data package with **20,000 synthetic pairs** that were NOT part of the original OpenAssistant or mock datasets:
  - 10,000 vague → strategic clarification questions
  - 10,000 clear → R.C.F.O formatted prompts
- `inspector/data/processed/expanded_dataset.json`
- `inspector/data/raw/expanded_dataset.json`

### How it works

The generator uses template banks over roles, contexts, focuses and outputs to create diverse pairs without depending on external APIs or proprietary models (safe for commercialization):

- **Vague prompts** (e.g., "fix my code", "help me with this project", "make it better") → mapped to the four strategic R.C.F.O-aligned clarification questions (Role, Context, Focus, Output).
- **Clear prompts** (e.g., "As a senior Python developer, refactor this legacy authentication module to use OAuth2 and return a JSON report...") → mapped to complete R.C.F.O structures.

### Files produced

- `scripts/expand_dataset.py`
- `inspector/data/processed/expanded_dataset.json`
- `inspector/data/raw/expanded_dataset.json`

### Recommended next step

Use this expanded dataset (or merge it with `openassistant_dataset.json`) to retrain the model on the Google Colab GPU with larger hyperparameters (D_MODEL=256, NUM_LAYERS=4, EPOCHS=20+) for commercial-quality generation.

## Session 17 — GPU Training Support (Colab-Ready)

### What was added

- `scripts/train.py` — rewritten with automatic device detection:
  - NVIDIA CUDA (priority 1)
  - Intel/AMD DirectML (priority 2, Windows)
  - CPU (fallback)
  - Environment variable `INSPECTOR_DEVICE` to force a specific device.
- Model and data tensors are moved to the selected device automatically.
- Hyperparameters exposed at the top of the file for easy scaling:
  - `VOCAB_SIZE`, `D_MODEL`, `NUM_HEADS`, `NUM_LAYERS`, `D_FF`, `MAX_LEN`,
    `BATCH_SIZE`, `EPOCHS`, `LR`, `MAX_PAIRS`.
- `scripts/colab_train.ipynb` — Google Colab notebook ready to run on the free T4 GPU:
  1. Upload `train.py`, `model.py`, `tokenizer.py`, `openassistant_dataset.json`
  2. Set runtime to GPU
  3. Run all cells
  4. Download `inspector_model.pth` and `tokenizer_vocab.json`

### Recommended hyperparameters for commercial quality

| Scenario | D_MODEL | NUM_LAYERS | NUM_HEADS | D_FF | EPOCHS | MAX_PAIRS | Expected time |
|----------|---------|------------|-----------|------|--------|-----------|---------------|
| CPU MVP (local) | 32 | 1 | 4 | 128 | 2 | 3.000 | 2–5 min |
| Colab T4 (fast) | 256 | 4 | 8 | 1024 | 20 | None (12k+) | 15–40 min |
| Colab T4 (quality) | 512 | 6 | 8 | 2048 | 30 | None (12k+) | 1–2 hours |

### How to train on Colab

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `scripts/train.py`, `scripts/model.py`, `scripts/tokenizer.py`, `inspector/data/processed/openassistant_dataset.json`
3. Runtime → Change runtime type → GPU
4. Run `scripts/colab_train.ipynb` cell by cell
5. Download the trained `inspector_model.pth` and `tokenizer_vocab.json`
6. Copy them back to `InspectorAI/inspector/models/` on your local machine
7. Run `python inspector.py` locally (inference is CPU-light)

### Important notes

- Your local machine has **Intel Iris Xe Graphics** (integrated, ~2 GB shared) and PyTorch is installed without CUDA. Local training beyond the small MVP will be slow. Use Colab for production-quality training.
- Inference (`inspector.py`) runs fine on CPU — no GPU needed to use the trained model.
- OpenAssistant `oasst1` is Apache 2.0 licensed — safe for commercialization.


## Session 18 — Legacy Code Archive (2026-11-03)

### What changed

- Created `legacy/` to preserve code that is no longer part of the supported
  training and inference workflow.
- Moved the previous heuristic package components:
  - `inspector/config.py`
  - `inspector/core/`
  - The duplicate historical test under `inspector/tests/`
- Moved retired data and training scripts to `legacy/scripts/`:
  - `train_colab.py`
  - `dataset_builder.py`
  - `expand_dataset.py`
  - `expand_dataset_60k.py`
  - `build_alpaca_dataset.py`
  - `build_openassistant_dataset.py`
- Added `legacy/README.md` to explain the archive and distinguish it from
  supported project commands.
- Updated the root README project tree.

### Supported workflow

The current supported workflow remains:

- `python scripts/prepare_dataset.py`
- `python scripts/train.py`
- `python -m inspector`
- `python inspector_gui.py`

Archived modules are retained for historical reference and are not maintained
or tested as part of the current application.
