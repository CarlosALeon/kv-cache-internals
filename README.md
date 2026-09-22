# kv-cache-from-scratch

Ejemplificacion practica del **KV cache** en modelos de lenguaje (LLMs),
implementado desde cero en PyTorch con fines educativos.

El objetivo no es una implementacion de produccion, sino entender el termino:
que es, por que funciona, que reto abre y que significa para la investigacion
en Machine Learning.

## Idea en una frase

Durante la generacion autorregresiva, los vectores key (K) y value (V) de los
tokens ya vistos no cambian entre pasos. El KV cache los guarda y reutiliza en
lugar de recalcularlos, bajando el costo por paso de O(n^2) a O(n) lineal, a
cambio de una memoria que crece linealmente con la longitud de la secuencia.

## Estructura del repositorio

```
kv-cache-from-scratch/
├── README.md              Este archivo
├── requirements.txt       Dependencias
├── LICENSE                Licencia MIT
├── .gitignore
├── notebooks/
│   └── kv_cache_para_ml_research.ipynb   Notebook explicativo y ejecutable
├── src/
│   └── kv_cache.py        Implementacion reutilizable + demo por linea de comandos
└── docs/
    └── conceptos.xml      Explicacion estructurada del concepto (formato plano)
```

## Requisitos

- Python 3.10 o superior
- PyTorch (ver `requirements.txt`)

No requiere GPU. Todo corre en CPU.

## Entorno local (paso a paso)

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/CarlosALeon/kv-cache-from-scratch.git
   cd kv-cache-from-scratch
   ```

2. Crear y activar un entorno virtual:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # En Windows: .venv\Scripts\activate
   ```

3. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Como ejecutar

### Opcion 1: script por linea de comandos

```bash
python src/kv_cache.py --tokens 200 --scaling
```

Salida esperada (los tiempos varian segun la maquina):

```
Correctitud (salidas identicas): True
Sin cache: 0.045 s
Con cache: 0.014 s
Speed-up:  3.29x

    n  sin_cache  con_cache  speed-up
   50      0.011      0.004     2.99x
  100      0.021      0.007     3.10x
  200      0.052      0.015     3.60x
  400      0.170      0.029     5.82x
```

La tabla muestra el punto clave: a mayor numero de tokens, mayor el speed-up.
Es la evidencia empirica del cambio de O(n^2) a O(n).

### Opcion 2: notebook

Abrir `notebooks/kv_cache_para_ml_research.ipynb` en Jupyter, VS Code o Google
Colab y ejecutar las celdas en orden. El notebook explica cada paso y su porque.

## Que se verifica

1. **Correctitud**: generar con y sin cache produce exactamente la misma
   secuencia. Una optimizacion que cambia el resultado es un bug, no una
   optimizacion.
2. **Speed-up**: el tiempo de generacion se reduce al reutilizar K/V.
3. **Escalamiento**: el beneficio crece con la longitud de la secuencia.

## Notas honestas

- El modelo del ejemplo no esta entrenado, por lo que genera texto sin
  sentido. Es irrelevante para el proposito: el KV cache es una optimizacion
  de inferencia y no cambia *que* se genera, solo *que tan rapido*.
- Las mediciones son en CPU. En GPU y con modelos pequenos el patron puede
  variar por el costo de transferencia y comunicacion.
- La implementacion prioriza claridad sobre eficiencia. Implementaciones
  reales usan preasignacion de memoria, ventanas deslizantes o gestion de
  memoria a nivel de sistema (por ejemplo, PagedAttention en vLLM).

## Referencias y atribucion

Este proyecto es una reimplementacion propia con fines educativos, basada en
el concepto explicado por Sebastian Raschka. No reproduce codigo ni texto
protegido; todo el credito conceptual es de las fuentes citadas.

- Sebastian Raschka (2025). *Understanding and Coding the KV Cache in LLMs
  from Scratch*.
  https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
- A Survey on LLM Acceleration based on KV Cache Management — arXiv:2412.19442
- KV Cache Optimization Strategies for Scalable and Efficient LLM Inference —
  arXiv:2603.20397
- KV Cache Compression for Inference Efficiency in LLMs: A Review —
  arXiv:2508.06297
- Multi-Query Attention (Shazeer, 2019) — arXiv:1911.02150
- Grouped-Query Attention (Ainslie et al., 2023) — arXiv:2305.13245
- H2O: Heavy-Hitter Oracle (2023) — arXiv:2306.14048

Ver `docs/conceptos.xml` para una explicacion estructurada adicional.

## Autor

Carlos Alberto León Gil — https://github.com/CarlosALeon
Ingeniero industrial. Estudiante de maestria en Ciencia de Datos.

## Licencia

MIT. Ver `LICENSE`.
