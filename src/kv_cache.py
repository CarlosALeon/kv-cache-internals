"""
kv_cache.py

Implementacion desde cero de un KV cache para atencion causal, con fines
educativos. El objetivo es ejemplificar el termino "KV cache" y su compromiso
computo-memoria, no reemplazar implementaciones de produccion.

Fuente conceptual: Sebastian Raschka, "Understanding and Coding the KV Cache
in LLMs from Scratch" (2025). Este codigo es una reimplementacion propia con
fines de estudio.

Uso por linea de comandos:
    python -m src.kv_cache --tokens 200
    python src/kv_cache.py --tokens 200 --scaling
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    """Atencion causal de una cabeza, con y sin KV cache.

    Idea central: en atencion causal, los vectores key (K) y value (V) de los
    tokens ya vistos no dependen de los tokens futuros. Son invariantes entre
    pasos de generacion, por lo que recalcularlos es trabajo redundante.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        # Buffers de cache: arrancan vacios.
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def reset_cache(self) -> None:
        # Entre dos prompts distintos hay que limpiar, o el modelo atenderia a
        # keys viejos y produciria salida incoherente.
        self.cache_k, self.cache_v = None, None

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        q = self.W_query(x)
        k_new = self.W_key(x)
        v_new = self.W_value(x)

        if use_cache:
            # Solo calculamos K/V del token nuevo; el resto se recupera.
            if self.cache_k is None:
                self.cache_k, self.cache_v = k_new, v_new
            else:
                self.cache_k = torch.cat([self.cache_k, k_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, v_new], dim=1)
            k, v = self.cache_k, self.cache_v
        else:
            # Sin cache: K/V se recalculan sobre toda la secuencia cada vez.
            k, v = k_new, v_new

        # Atencion causal enmascarada.
        scores = q @ k.transpose(-2, -1) / (self.d_out ** 0.5)
        Tq, Tk = q.shape[1], k.shape[1]
        offset = Tk - Tq  # posicion absoluta desde donde arrancan los queries
        mask = torch.triu(
            torch.ones(Tq, Tk, device=x.device, dtype=torch.bool),
            diagonal=1 + offset,
        )
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return attn @ v


class MiniGPT(nn.Module):
    """Modelo minimo autorregresivo para ilustrar el flujo de generacion."""

    def __init__(self, vocab_size: int, d_model: int, ctx_len: int) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        self.att = CausalAttention(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.ctx_len = ctx_len
        self.current_pos = 0  # cuantos tokens ya cacheamos

    def reset_cache(self) -> None:
        self.att.reset_cache()
        self.current_pos = 0

    def forward(self, idx: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        _, T = idx.shape
        if use_cache:
            # Las posiciones nuevas continuan donde quedamos.
            pos = torch.arange(self.current_pos, self.current_pos + T, device=idx.device)
            self.current_pos += T
        else:
            pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos).unsqueeze(0)
        x = self.att(x, use_cache=use_cache)
        return self.head(x)


@torch.no_grad()
def generate(model: MiniGPT, idx: torch.Tensor, max_new_tokens: int, use_cache: bool) -> torch.Tensor:
    """Genera tokens de forma greedy (argmax) con o sin KV cache."""
    model.eval()
    if use_cache:
        model.reset_cache()
        logits = model(idx, use_cache=True)  # prefill: procesa el prompt una vez
        for _ in range(max_new_tokens):
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_idx], dim=1)
            logits = model(next_idx, use_cache=True)  # decode: solo el token nuevo
    else:
        for _ in range(max_new_tokens):
            logits = model(idx[:, -model.ctx_len:], use_cache=False)  # reprocesa todo
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_idx], dim=1)
    return idx


def run_demo(n_tokens: int = 200, seed: int = 123, scaling: bool = False) -> None:
    torch.manual_seed(seed)
    vocab_size, d_model, ctx_len = 100, 64, 512
    model = MiniGPT(vocab_size, d_model, ctx_len)
    prompt = torch.randint(0, vocab_size, (1, 4))

    # 1) Correctitud: con y sin cache deben producir la misma secuencia.
    out_no = generate(model, prompt.clone(), n_tokens, use_cache=False)
    out_yes = generate(model, prompt.clone(), n_tokens, use_cache=True)
    same = torch.equal(out_no, out_yes)
    print(f"Correctitud (salidas identicas): {same}")
    if not same:
        raise AssertionError("El KV cache cambio el resultado: hay un bug de indexacion.")

    # 2) Speed-up.
    def timeit(use_cache: bool, reps: int = 3) -> float:
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            generate(model, prompt.clone(), n_tokens, use_cache)
            best = min(best, time.perf_counter() - t0)
        return best

    t_no = timeit(False)
    t_yes = timeit(True)
    print(f"Sin cache: {t_no:.3f} s")
    print(f"Con cache: {t_yes:.3f} s")
    print(f"Speed-up:  {t_no / t_yes:.2f}x")

    # 3) Escalamiento O(n^2) -> O(n): a mayor n, mayor speed-up.
    if scaling:
        def run(uc: bool, n: int) -> float:
            t0 = time.perf_counter()
            generate(model, prompt.clone(), n, uc)
            return time.perf_counter() - t0

        print()
        print(f"{'n':>5} {'sin_cache':>10} {'con_cache':>10} {'speed-up':>9}")
        for n in [50, 100, 200, 400]:
            a = min(run(False, n) for _ in range(2))
            b = min(run(True, n) for _ in range(2))
            print(f"{n:5d} {a:10.3f} {b:10.3f} {a / b:8.2f}x")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo educativa del KV cache desde cero.")
    p.add_argument("--tokens", type=int, default=200, help="tokens nuevos a generar")
    p.add_argument("--seed", type=int, default=123, help="semilla de reproducibilidad")
    p.add_argument("--scaling", action="store_true", help="mostrar tabla de escalamiento")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_demo(n_tokens=args.tokens, seed=args.seed, scaling=args.scaling)
