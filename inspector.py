"""Inspector AI — Interactive Application.

Loads the trained InspectorTransformer and the from-scratch BPE tokenizer,
then runs an interactive terminal loop:

  You:    <types an idea / prompt>
  Inspector:  <encoded -> Transformer -> decoded response>

The model output will naturally be either:
  * strategic clarification questions  (if the input was dubious / vague), or
  * a strict R.C.F.O formatted prompt  (if the input was clear enough).

No HuggingFace, no sentencepiece — everything is built from torch primitives
and the custom tokenizer trained in previous sessions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[0]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tokenizer import BPETokenizer, SOS_IDX, EOS_IDX, PAD_IDX  # noqa: E402
from model import InspectorTransformer  # noqa: E402

MODEL_PATH = ROOT / "inspector" / "models" / "inspector_model.pth"
TOKENIZER_PATH = ROOT / "inspector" / "models" / "tokenizer_vocab.json"

# Inference hyperparameters (fallback defaults; overridden by state_dict)
# Valores de fallback para checkpoints antigos.
# Checkpoints novos usam a configuração salva durante o treinamento.
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 512
MAX_LEN = 128
MAX_NEW_TOKENS = 40

# Beam search hyperparameters
BEAM_SIZE = 5           # number of candidate sequences to keep at each step
LENGTH_PENALTY = 0.6    # <1.0 favors shorter sequences, >1.0 favors longer ones


def _infer_model_dims(state_dict: dict) -> dict:
    """
    Recupera as dimensões que podem ser identificadas pelos pesos.

    Atenção:
        O número de cabeças não pode ser inferido diretamente
        pelos pesos dessa implementação de MultiHeadAttention.
        Para checkpoints antigos, utiliza-se o maior divisor
        de d_model que não ultrapasse 8.
    """

    # ---------------------------------------------------------------
    # Dimensões dos embeddings
    # ---------------------------------------------------------------
    emb_weight = state_dict.get("encoder.embedding.weight")

    if emb_weight is None:
        raise ValueError(
            "Checkpoint inválido: embedding do encoder não encontrado."
        )

    vocab_size = int(emb_weight.shape[0])
    d_model = int(emb_weight.shape[1])

    # ---------------------------------------------------------------
    # Número de camadas do encoder
    # ---------------------------------------------------------------
    encoder_layers = 0

    while (
        f"encoder.layers.{encoder_layers}.self_attn.q_proj.weight"
        in state_dict
    ):
        encoder_layers += 1

    # ---------------------------------------------------------------
    # Número de camadas do decoder
    # ---------------------------------------------------------------
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
    # Dimensão feed-forward
    # ---------------------------------------------------------------
    ffn_weight = state_dict.get("encoder.layers.0.ffn.0.weight")

    if ffn_weight is None:
        raise ValueError(
            "Não foi possível identificar a dimensão feed-forward."
        )

    d_ff = int(ffn_weight.shape[0])

    # ---------------------------------------------------------------
    # Comprimento máximo de sequência
    # ---------------------------------------------------------------
    # Sempre prioriza o max_len salvo no checkpoint (pos_enc.pe).
    # Isso evita mismatch entre o modelo treinado e o modelo inferido.
    positional_encoding = state_dict.get("encoder.pos_enc.pe")

    if positional_encoding is not None:
        max_len = int(positional_encoding.shape[1])
    else:
        # Fallback: tenta ler do model_config, depois do MAX_LEN hardcoded
        max_len = int(saved_config.get("max_len", MAX_LEN)) if saved_config else MAX_LEN

    # ---------------------------------------------------------------
    # Número de cabeças
    # ---------------------------------------------------------------
    # Em checkpoints antigos, essa informação não está registrada.
    num_heads = min(NUM_HEADS, d_model)

    while num_heads > 1 and d_model % num_heads != 0:
        num_heads -= 1

    if d_model % num_heads != 0:
        num_heads = 1

    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "max_len": max_len,
    }


def load_components() -> tuple:
    """
    Carrega tokenizer e modelo treinado.

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
    # 1. Carrega tokenizer
    # ---------------------------------------------------------------
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    print(
        f"Tokenizer carregado: {len(tokenizer.vocab)} tokens."
    )

    # ---------------------------------------------------------------
    # 2. Carrega checkpoint
    # ---------------------------------------------------------------
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    is_new_checkpoint = (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    )

    if is_new_checkpoint:
        state_dict = checkpoint["model_state_dict"]

        saved_config = checkpoint.get("model_config", {})

        # Primeiro recupera o que for possível pelos pesos.
        dims = _infer_model_dims(state_dict)

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
    src: torch.Tensor,
    src_mask: torch.Tensor,
    enc_out: torch.Tensor,
) -> List[int]:
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
        return []

    # Retorna o melhor resultado, removendo SOS.
    best_tokens, _, _ = beams[0]

    return best_tokens[1:]


def generate_response(
    model: InspectorTransformer,
    tokenizer: BPETokenizer,
    user_input: str,
) -> str:
    """
    Gera uma resposta para uma entrada do usuário.

    Executa o encoder uma vez e utiliza Beam Search
    para a decodificação autoregressiva.
    """

    model.eval()

    user_input = user_input.strip()

    if not user_input:
        return "I did not understand that. Please provide a prompt."

    # Recupera o limite posicional real do modelo.
    model_max_len = int(model.encoder.pos_enc.pe.shape[1])

    # Codifica e limita a entrada.
    src_ids = tokenizer.encode(user_input)

    src_ids = src_ids[:model_max_len]

    if not src_ids:
        return "I did not understand that. Please try again."

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
    )

    src_mask = InspectorTransformer.create_padding_mask(
        src,
        PAD_IDX,
    )

    with torch.no_grad():

        # Executa o encoder uma única vez.
        enc_out = model.encode(
            src,
            src_mask,
        )

        # Decodifica por Beam Search.
        generated_ids = _beam_search(
            model,
            tokenizer,
            src,
            src_mask,
            enc_out,
        )

    response = tokenizer.decode(
        generated_ids,
        skip_special=True,
    )

    if not response.strip():
        return "I could not generate a response. Please try again."

    return response.strip()