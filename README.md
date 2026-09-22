# kv-cache-internals

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CarlosALeon/kv-cache-internals/blob/main/notebooks/kv_cache_mechanics.ipynb)
[![CI](https://github.com/CarlosALeon/kv-cache-internals/actions/workflows/run.yml/badge.svg)](https://github.com/CarlosALeon/kv-cache-internals/actions/workflows/run.yml)

A from-scratch look at how the KV cache works inside a transformer, and what
that mechanism costs you. Runnable in the browser, no local install required.

*Una mirada desde cero a como funciona el KV cache dentro de un transformer, y
que cuesta ese mecanismo. Ejecutable en el navegador, sin instalar nada.*

---

## EN

### What this is

A minimal, readable implementation of key-value caching for causal attention.
It generates the same tokens with and without the cache, times both, and shows
how the gap widens as sequences grow. The point is the mechanism, not a
production kernel.

### What the KV cache does

During autoregressive decoding the key (K) and value (V) vectors of past tokens
do not change from step to step. Recomputing them every step is wasted work.
The cache stores them once and reuses them. Per-step cost drops from quadratic
to linear; memory grows linearly with sequence length. That trade is the whole
story.

### When to use it

- Autoregressive inference, one token at a time, where you re-read a growing
  prefix on every step.
- Long generations or long prompts, where the recomputation you avoid keeps
  adding up.
- Serving, where latency per token is what users feel.

### When not to use it

- Training. The full sequence is processed in parallel with teacher forcing;
  there is nothing to reuse.
- A single forward pass over a fixed sequence (classification, embeddings,
  scoring). No incremental decoding, no cache.
- Memory-bound settings with very long contexts, unless you also compress,
  evict, or share the cache. See limits below.

### Why it matters

Whether you can serve a 4K or a 128K context, how much GPU memory you need, and
what a token costs, all come back to how you manage this cache. It is the point
where a clean idea meets a hard systems constraint.

### Limits and boundaries

- Memory is the ceiling. The cache grows with every token; for large models and
  long contexts it can exceed the model's own footprint.
- Naive `torch.cat` growth fragments memory and reallocates. Real systems
  preallocate or page it (PagedAttention / vLLM).
- Sharing K/V across heads (MQA, GQA, MLA) shrinks the cache but is an
  architecture decision, not a drop-in switch.
- Eviction and compression trade recall for memory; they can drop tokens you
  later needed.
- This repo runs on CPU and favors clarity. On GPU with tiny models, transfer
  and launch overhead can outweigh the savings.

### Run it

Zero install: click the Colab badge above.

Local:

```bash
git clone https://github.com/CarlosALeon/kv-cache-internals.git
cd kv-cache-internals
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/kv_cache.py --tokens 200 --scaling
```

Expected shape of the output (numbers vary by machine):

```
Correctitud (salidas identicas): True
Speed-up:  ~3x
n grows -> speed-up grows   (empirical O(n^2) -> O(n))
```

### Layout

```
notebooks/kv_cache_mechanics.ipynb   walkthrough + runnable cells
src/kv_cache.py                             reusable implementation + CLI
docs/conceptos.xml                          structured notes (what/when/why/limits)
```

---

## ES

### Que es

Una implementacion minima y legible del key-value caching para atencion causal.
Genera los mismos tokens con y sin cache, mide ambos, y muestra como la
diferencia crece con la longitud de la secuencia. Importa el mecanismo, no un
kernel de produccion.

### Que hace el KV cache

En decodificacion autorregresiva, los vectores key (K) y value (V) de los
tokens pasados no cambian entre pasos. Recalcularlos en cada paso es trabajo
perdido. El cache los guarda una vez y los reutiliza. El costo por paso baja de
cuadratico a lineal; la memoria crece linealmente con la longitud. Ese
compromiso es toda la historia.

### Cuando usarlo

- Inferencia autorregresiva, token por token, donde relees un prefijo que
  crece en cada paso.
- Generaciones largas o prompts largos, donde el recomputo que evitas se
  acumula.
- Servir modelos, donde la latencia por token es lo que siente el usuario.

### Cuando no usarlo

- Entrenamiento. La secuencia se procesa en paralelo con teacher forcing; no
  hay nada que reutilizar.
- Un solo forward sobre una secuencia fija (clasificacion, embeddings, scoring).
  Sin decodificacion incremental, no hay cache.
- Escenarios limitados por memoria con contextos muy largos, salvo que ademas
  comprimas, descartes (eviction) o compartas el cache. Ver limites.

### Por que importa

Poder servir un contexto de 4K o de 128K, cuanta memoria de GPU necesitas y
cuanto cuesta un token, todo vuelve a como gestionas este cache. Es el punto
donde una idea limpia choca con una restriccion real de sistemas.

### Limites y fronteras

- La memoria es el techo. El cache crece con cada token; en modelos grandes y
  contextos largos puede superar el tamano del propio modelo.
- El crecimiento con `torch.cat` fragmenta y reasigna memoria. Los sistemas
  reales preasignan o paginan (PagedAttention / vLLM).
- Compartir K/V entre cabezas (MQA, GQA, MLA) reduce el cache, pero es una
  decision de arquitectura, no un interruptor.
- Eviction y compresion cambian memoria por capacidad de recuerdo; pueden
  descartar tokens que luego necesitabas.
- Este repo corre en CPU y prioriza claridad. En GPU con modelos pequenos, el
  costo de transferencia y lanzamiento puede superar el ahorro.

### Ejecutar

Sin instalar: clic en el badge de Colab arriba.

Local: mismos comandos de la seccion EN.

---

## References / Referencias

Reimplementation for study, with attribution. No protected code or text is
reproduced. *Reimplementacion de estudio, con atribucion. No se reproduce
codigo ni texto protegido.*

- Sebastian Raschka (2025), *Understanding and Coding the KV Cache in LLMs from
  Scratch* — https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
- A Survey on LLM Acceleration based on KV Cache Management — arXiv:2412.19442
- KV Cache Optimization Strategies for Scalable and Efficient LLM Inference — arXiv:2603.20397
- KV Cache Compression for Inference Efficiency in LLMs: A Review — arXiv:2508.06297
- Multi-Query Attention (Shazeer, 2019) — arXiv:1911.02150
- Grouped-Query Attention (Ainslie et al., 2023) — arXiv:2305.13245
- H2O: Heavy-Hitter Oracle (2023) — arXiv:2306.14048

## Author / Autor

Carlos Alberto León Gil — https://github.com/CarlosALeon

## License / Licencia

MIT. See `LICENSE`.
