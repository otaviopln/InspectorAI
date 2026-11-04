"""Central inference engine for InspectorAI.

This module owns checkpoint loading, architecture validation and autoregressive
response generation. It is imported by both the desktop GUI and terminal CLI.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from scripts.model import InspectorTransformer
from scripts.tokenizer import BPETokenizer, EOS_IDX, PAD_IDX, SOS_IDX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "inspector" / "models" / "inspector_model.pth"
TOKENIZER_PATH = PROJECT_ROOT / "inspector" / "models" / "tokenizer_vocab.json"
DATASET_PATH = PROJECT_ROOT / "inspector" / "data" / "processed" / "prepared_dataset.json"

# Inference hyperparameters (fallback defaults; overridden by state_dict)
# Valores de fallback para checkpoints antigos.
# Checkpoints novos usam a configuração salva durante o treinamento.
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 512
MAX_LEN = 128
MAX_NEW_TOKENS = 191

# Beam search hyperparameters
BEAM_SIZE = 5           # number of candidate sequences to keep at each step
LENGTH_PENALTY = 0.6    # <1.0 favors shorter sequences, >1.0 favors longer ones

HASH_ALGORITHM = "sha256-canonical-json-v1"


def _canonical_json_sha256(path: Path) -> str:
    """Hash a JSON artifact independently of indentation and line endings."""
    if not path.is_file():
        raise FileNotFoundError(f"Artifact JSON necessário à validação não encontrado: {path}")

    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    canonical = json.dumps(
        data,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_checkpoint_artifact_hashes(
    checkpoint: dict,
    tokenizer_path: Path = TOKENIZER_PATH,
    dataset_path: Path = DATASET_PATH,
    validate_dataset: bool = True,
) -> bool:
    """Validate hashes; dataset validation is required for reproducible evaluation."""
    artifact_hashes = checkpoint.get("artifact_hashes")
    format_version = int(checkpoint.get("format_version", 0))

    if artifact_hashes is None:
        if format_version >= 4:
            raise ValueError(
                "Checkpoint v4 inválido: metadados artifact_hashes ausentes."
            )
        print(
            "AVISO: checkpoint legado sem hashes de integridade. "
            "Não é possível confirmar se tokenizer e dataset correspondem "
            "ao treinamento original."
        )
        return False

    if not isinstance(artifact_hashes, dict):
        raise ValueError("Checkpoint inválido: artifact_hashes precisa ser um dicionário.")

    algorithm = artifact_hashes.get("algorithm")
    if algorithm != HASH_ALGORITHM:
        raise ValueError(
            "Algoritmo de hash incompatível no checkpoint: "
            f"{algorithm!r}. Esperado {HASH_ALGORITHM!r}."
        )

    artifacts = [("tokenizer", tokenizer_path)]
    if validate_dataset:
        artifacts.append(("prepared_dataset", dataset_path))

    for artifact_name, artifact_path in artifacts:
        expected_hash = artifact_hashes.get(artifact_name)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(
                f"Checkpoint inválido: hash SHA-256 de {artifact_name} ausente ou inválido."
            )

        actual_hash = _canonical_json_sha256(Path(artifact_path))
        if actual_hash != expected_hash:
            raise ValueError(
                f"Falha na validação de integridade de {artifact_name}: "
                f"hash do checkpoint={expected_hash}; hash local={actual_hash}. "
                "Os artefatos não pertencem ao mesmo treinamento. "
                "Restaure os arquivos correspondentes ou treine novamente."
            )

    validated = "tokenizer OK"
    if validate_dataset:
        validated += " | dataset OK"
    print(f"Integridade dos artefatos validada (SHA-256 canônico): {validated}.")
    return True


def _infer_model_dims(
    state_dict: dict,
    saved_config: dict | None = None,
) -> dict:
    """
    Recupera as dimensões da arquitetura pelos pesos e,
    quando disponível, pela configuração salva no checkpoint.

    O número de cabeças não pode ser recuperado diretamente
    dos pesos da MultiHeadAttention. Para checkpoints novos,
    utiliza a configuração salva. Para checkpoints antigos,
    utiliza NUM_HEADS como estimativa.
    """

    if saved_config is None:
        saved_config = {}

    if not isinstance(saved_config, dict):
        raise ValueError(
            "Configuração inválida no checkpoint: "
            "model_config precisa ser um dicionário."
        )

    # ---------------------------------------------------------------
    # 1. Dimensões dos embeddings
    # ---------------------------------------------------------------
    emb_weight = state_dict.get("encoder.embedding.weight")

    if emb_weight is None:
        raise ValueError(
            "Checkpoint inválido: embedding do encoder não encontrado."
        )

    vocab_size = int(emb_weight.shape[0])
    d_model = int(emb_weight.shape[1])

    # ---------------------------------------------------------------
    # 2. Número de camadas
    # ---------------------------------------------------------------
    encoder_layers = 0

    while (
        f"encoder.layers.{encoder_layers}.self_attn.q_proj.weight"
        in state_dict
    ):
        encoder_layers += 1

    decoder_layers = 0

    while (
        f"decoder.layers.{decoder_layers}.masked_self_attn.q_proj.weight"
        in state_dict
    ):
        decoder_layers += 1

    if encoder_layers == 0 or decoder_layers == 0:
        raise ValueError(
            "Não foi possível identificar as camadas do Transformer."
        )

    if encoder_layers != decoder_layers:
        raise ValueError(
            "Arquitetura incompatível: encoder e decoder possuem "
            "quantidades diferentes de camadas."
        )

    num_layers = encoder_layers

    # ---------------------------------------------------------------
    # 3. Dimensão feed-forward
    # ---------------------------------------------------------------
    ffn_weight = state_dict.get(
        "encoder.layers.0.ffn.0.weight"
    )

    if ffn_weight is None:
        raise ValueError(
            "Não foi possível identificar a dimensão feed-forward."
        )

    d_ff = int(ffn_weight.shape[0])

    # ---------------------------------------------------------------
    # 4. Comprimento máximo de sequência
    # ---------------------------------------------------------------
    encoder_pe = state_dict.get("encoder.pos_enc.pe")
    decoder_pe = state_dict.get("decoder.pos_enc.pe")

    positional_buffers = [
        buffer
        for buffer in (encoder_pe, decoder_pe)
        if buffer is not None
    ]

    if positional_buffers:
        lengths = {
            int(buffer.shape[1])
            for buffer in positional_buffers
        }

        if len(lengths) != 1:
            raise ValueError(
                "Encoder e decoder possuem comprimentos "
                "posicionais incompatíveis."
            )

        max_len = lengths.pop()

    else:
        # Compatibilidade com checkpoints antigos.
        # Não existe buffer posicional para recuperar esse valor.
        max_len = int(
            saved_config.get("max_len", MAX_LEN)
        )

    # ---------------------------------------------------------------
    # 5. Número de cabeças
    # ---------------------------------------------------------------
    if "num_heads" in saved_config:
        num_heads = int(saved_config["num_heads"])
    else:
        # Checkpoints antigos não armazenam essa informação.
        num_heads = min(NUM_HEADS, d_model)

        while num_heads > 1 and d_model % num_heads != 0:
            num_heads -= 1

    if num_heads <= 0 or d_model % num_heads != 0:
        raise ValueError(
            f"Configuração inválida: d_model={d_model} "
            f"não é divisível por num_heads={num_heads}."
        )

    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "max_len": max_len,
    }



def load_components(validate_dataset: bool = False) -> tuple:
    """
    Carrega tokenizer e modelo treinado.

    A inferência valida o tokenizador, mas não depende do dataset de treino.
    A avaliação reproduzível deve usar load_components(validate_dataset=True).

    Suporta:
        1. Checkpoints novos com configuração explícita.
        2. Checkpoints antigos contendo somente state_dict.

    Retorna:
        (tokenizer, model)
    """

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer não encontrado em {TOKENIZER_PATH}. "
            "Execute scripts/train.py primeiro."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Execute scripts/train.py primeiro."
        )

    # ---------------------------------------------------------------
    # 1. Carrega checkpoint e valida os artefatos associados
    # ---------------------------------------------------------------
    # Load only checkpoints produced by this project or another trusted source.
    # requirements.txt sets the minimum version with known serialization fixes.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    is_new_checkpoint = (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    )

    if is_new_checkpoint:
        _validate_checkpoint_artifact_hashes(
            checkpoint,
            validate_dataset=validate_dataset,
        )

    # Load the tokenizer only after its fingerprint has been verified.
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    print(f"Tokenizer carregado: {len(tokenizer.vocab)} tokens.")

    if is_new_checkpoint:
        state_dict = checkpoint["model_state_dict"]

        saved_config = checkpoint.get("model_config") or {}

        if not isinstance(saved_config, dict):
            raise ValueError(
                "Checkpoint inválido: model_config precisa ser um dicionário."
            )

        # Recupera as dimensões usando os pesos e a configuração salva.
        dims = _infer_model_dims(
            state_dict,
            saved_config,
        )

        # Depois sobrescreve com a configuração exata salva.
        config_keys = (
            "vocab_size",
            "d_model",
            "num_heads",
            "num_layers",
            "d_ff",
            "max_len",
        )

        for key in config_keys:
            if key in saved_config:
                dims[key] = int(saved_config[key])

        print("Checkpoint novo reconhecido.")

    else:
        # Compatibilidade com checkpoints antigos.
        state_dict = checkpoint
        saved_config = {}
        dims = _infer_model_dims(state_dict)

        print(
            "AVISO: checkpoint antigo detectado. "
            "O número de cabeças de atenção foi estimado."
        )

    # ---------------------------------------------------------------
    # 3. Valida configuração
    # ---------------------------------------------------------------
    if dims["d_model"] % dims["num_heads"] != 0:
        raise ValueError(
            "Configuração inválida: d_model precisa ser divisível "
            "por num_heads."
        )

    if dims["max_len"] < 2:
        raise ValueError("max_len inválido no checkpoint.")

    tokenizer_max_id = max(tokenizer.vocab.keys(), default=-1)

    if tokenizer_max_id >= dims["vocab_size"]:
        raise ValueError(
            "O vocabulário do tokenizer contém IDs que não existem "
            "na camada de embeddings do modelo. "
            "Treine novamente o modelo com o tokenizer correspondente."
        )

    # Checkpoints novos precisam ter correspondência exata.
    if (
        is_new_checkpoint
        and len(tokenizer.vocab) != dims["vocab_size"]
    ):
        raise ValueError(
            "O vocabulário do tokenizer não corresponde ao tamanho "
            "registrado no checkpoint. Verifique os arquivos do modelo "
            "e do tokenizer."
        )

    if (
        not is_new_checkpoint
        and len(tokenizer.vocab) != dims["vocab_size"]
    ):
        print(
            "AVISO: o checkpoint antigo possui um tamanho de "
            "vocabulário diferente do tokenizer. Tokens inexistentes "
            "no tokenizer serão bloqueados durante a geração."
        )

    print(
        "Model configuration: "
        f"vocab={dims['vocab_size']}, "
        f"d_model={dims['d_model']}, "
        f"heads={dims['num_heads']}, "
        f"layers={dims['num_layers']}, "
        f"d_ff={dims['d_ff']}, "
        f"max_len={dims['max_len']}"
    )

    # ---------------------------------------------------------------
    # 4. Reconstrói o modelo
    # ---------------------------------------------------------------
    model = InspectorTransformer(
        vocab_size=dims["vocab_size"],
        d_model=dims["d_model"],
        num_heads=dims["num_heads"],
        num_layers=dims["num_layers"],
        d_ff=dims["d_ff"],
        max_len=dims["max_len"],
        pad_idx=PAD_IDX,
    )

    # Carrega os pesos.
    model.load_state_dict(state_dict)

    model.eval()

    print(f"Model weights loaded from {MODEL_PATH.name}")

    return tokenizer, model

def _topk_logits(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retorna os k tokens mais prováveis e suas log-probabilidades.

    Args:
        logits: Tensor com as pontuações dos tokens.
        k: Quantidade máxima de candidatos.

    Returns:
        Tupla contendo:
            - Log-probabilidades dos candidatos.
            - IDs dos tokens selecionados.
    """

    if logits.dim() != 1:
        raise ValueError(
            "logits precisa ser um tensor unidimensional."
        )

    if k <= 0:
        raise ValueError(
            "k precisa ser maior que zero."
        )

    # Evita solicitar mais candidatos que o vocabulário disponível.
    k = min(k, logits.size(-1))

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    top_log_probs, top_indices = torch.topk(
        log_probs,
        k,
        dim=-1,
    )

    return top_log_probs, top_indices

def _beam_search(
    model: InspectorTransformer,
    tokenizer: BPETokenizer,
    src_mask: torch.Tensor,
    enc_out: torch.Tensor,
) -> tuple[List[int], bool]:
    """
    Decodificação autoregressiva com Beam Search.

    Restrições:
        - Não permite gerar PAD ou SOS.
        - Não permite gerar EOS no primeiro passo.
        - Não permite IDs inexistentes no tokenizer.
        - Respeita o comprimento máximo posicional do decoder.
    """

    # Cada beam:
    # (sequência de tokens, log_prob acumulado, finalizado)
    beams: List[Tuple[List[int], float, bool]] = [
        ([SOS_IDX], 0.0, False)
    ]

    # Impede a geração de sequências maiores que o PositionalEncoding.
    decoder_max_len = int(model.decoder.pos_enc.pe.shape[1])

    max_steps = min(
        MAX_NEW_TOKENS,
        max(decoder_max_len - 1, 0),
    )

    # IDs disponíveis no tokenizer.
    max_token_id = max(tokenizer.vocab.keys(), default=EOS_IDX)
    valid_vocab_size = max_token_id + 1

    for step in range(max_steps):
        candidates: List[Tuple[List[int], float, bool]] = []

        for tokens, log_prob, finished in beams:

            if finished:
                candidates.append((tokens, log_prob, True))
                continue

            tgt = torch.tensor(
                [tokens],
                dtype=torch.long,
            )

            # Máscara causal.
            causal_mask = InspectorTransformer.create_causal_mask(
                tgt.size(1),
                tgt.device,
            )

            tgt_pad_mask = InspectorTransformer.create_padding_mask(
                tgt,
                PAD_IDX,
            )

            tgt_mask = tgt_pad_mask & causal_mask

            logits = model.decode(
                tgt,
                enc_out,
                src_mask,
                tgt_mask,
            )

            next_logits = logits[0, -1, :].clone()

            # Bloqueia IDs que não existem no tokenizer.
            if valid_vocab_size < next_logits.size(0):
                next_logits[valid_vocab_size:] = float("-inf")

            # Não gera tokens de controle.
            next_logits[PAD_IDX] = float("-inf")
            next_logits[SOS_IDX] = float("-inf")

            # Obriga o modelo a produzir pelo menos um token de conteúdo.
            if step == 0:
                next_logits[EOS_IDX] = float("-inf")

            # Quantidade de tokens que ainda podem ser candidatos.
            allowed_count = (
                valid_vocab_size
                - 2
                - (1 if step == 0 else 0)
            )

            if allowed_count <= 0:
                continue

            k = min(BEAM_SIZE, allowed_count)

            top_log_probs, top_indices = _topk_logits(
                next_logits,
                k,
            )

            for i in range(k):
                token_id = int(top_indices[i].item())
                token_log_prob = float(top_log_probs[i].item())

                new_tokens = tokens + [token_id]
                new_log_prob = log_prob + token_log_prob

                is_finished = token_id == EOS_IDX

                candidates.append(
                    (
                        new_tokens,
                        new_log_prob,
                        is_finished,
                    )
                )

        if not candidates:
            break

        # Ordena pelo score com penalidade de comprimento.
        def _score(
            candidate: Tuple[List[int], float, bool]
        ) -> float:
            tokens, score, _ = candidate

            length = max(len(tokens) - 1, 1)

            return score / (length ** LENGTH_PENALTY)

        candidates.sort(
            key=_score,
            reverse=True,
        )

        beams = candidates[:BEAM_SIZE]

        if all(finished for _, _, finished in beams):
            break

    if not beams:
        return [], False

    # EOS marks a complete response; reaching max_steps does not.
    best_tokens, _, was_finished = beams[0]
    return best_tokens[1:], was_finished


@dataclass(frozen=True)
class InspectorResult:
    """Structured inference result; distinguishes errors from generated text."""

    status: str
    text: str
    input_tokens: int
    max_input_tokens: int
    finished: bool
    stop_reason: str


def generate_result(
    model: InspectorTransformer,
    tokenizer: BPETokenizer,
    user_input: str,
) -> InspectorResult:
    """Generate a response without silently truncating the user's input."""
    model.eval()
    user_input = user_input.strip()

    if not user_input:
        return InspectorResult(
            status="empty_input",
            text="Please provide a prompt.",
            input_tokens=0,
            max_input_tokens=0,
            finished=False,
            stop_reason="empty_input",
        )

    model_max_len = int(model.encoder.pos_enc.pe.shape[1])
    src_ids = tokenizer.encode(user_input)
    input_tokens = len(src_ids)

    if input_tokens > model_max_len:
        return InspectorResult(
            status="input_too_long",
            text=(
                f"Input too long: {input_tokens} tokens; "
                f"the model limit is {model_max_len}. "
                "Shorten the prompt; no content was discarded."
            ),
            input_tokens=input_tokens,
            max_input_tokens=model_max_len,
            finished=False,
            stop_reason="input_too_long",
        )

    if not src_ids:
        return InspectorResult(
            status="empty_input",
            text="Please provide a non-empty prompt.",
            input_tokens=0,
            max_input_tokens=model_max_len,
            finished=False,
            stop_reason="empty_input",
        )

    src = torch.tensor([src_ids], dtype=torch.long)
    src_mask = InspectorTransformer.create_padding_mask(src, PAD_IDX)

    with torch.no_grad():
        enc_out = model.encode(src, src_mask)
        generated_ids, was_finished = _beam_search(
            model,
            tokenizer,
            src_mask,
            enc_out,
        )

    response = tokenizer.decode(generated_ids, skip_special=True).strip()
    if not response:
        return InspectorResult(
            status="no_output",
            text="InspectorAI could not generate a response.",
            input_tokens=input_tokens,
            max_input_tokens=model_max_len,
            finished=was_finished,
            stop_reason="no_output",
        )

    if not was_finished:
        return InspectorResult(
            status="incomplete_generation",
            text=response,
            input_tokens=input_tokens,
            max_input_tokens=model_max_len,
            finished=False,
            stop_reason="max_tokens_reached",
        )

    return InspectorResult(
        status="ok",
        text=response,
        input_tokens=input_tokens,
        max_input_tokens=model_max_len,
        finished=True,
        stop_reason="eos_reached",
    )


def generate_response(
    model: InspectorTransformer,
    tokenizer: BPETokenizer,
    user_input: str,
) -> str:
    """Backward-compatible string API; raises for non-successful inference."""
    result = generate_result(model, tokenizer, user_input)
    if result.status != "ok":
        raise ValueError(result.text)
    return result.text
