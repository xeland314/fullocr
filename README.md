# fullocr — OCR + Traducción basado en Unlimited-OCR

Pipeline para PDFs escaneados que genera **versión original en inglés** (OCR) y **versión traducida al español**. Inspirado en [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) (One-shot Long-horizon Parsing), con fallback CPU para equipos sin CUDA (AMD Radeon).

> **Caso de uso inicial:** `REANIMACION NEONATAL CAPI 1-4.pdf` — 92 páginas escaneadas (sin texto seleccionable, 27 imágenes/página). Se genera `outputs/*_original.md` / `*_en.md` y `*_es.md`.

## Características

- **Unlimited-OCR cuando hay GPU NVIDIA:** si `--use-unlimited` y `torch.cuda.is_available()`, invoca `baidu/Unlimited-OCR` vía `transformers` (`model.infer_multi` con prompt `<image>Multi page parsing.`, `image_size=1024`, como en `infer.py:94`). Requiere CUDA 12.9/13.0 y `SGLang` (ver README original).
- **Fallback CPU para AMD / sin CUDA:** `PyMuPDF` (render DPI 200) + `Tesseract 5.5` (`--oem 1 --psm 1`, `lang=eng`) + `RapidOCR` opcional. Probado en Windows 11 + AMD Radeon Graphics + 8GB RAM.
- **Traducción con fallbacks locales:**
  1. `google` — `deep-translator` (GoogleTranslator, requiere internet, rápido)
  2. `opus` — `Helsinki-NLP/opus-mt-en-es` vía `transformers` + `torch` CPU (offline, 300MB)
  3. `llama` — GGUF vía `llama-cpp-python` (offline, generativo, `Qwen2.5-0.5B` / `TinyLlama` ~400MB-1GB)
  - Orden `auto: google -> opus -> llama` (configurable con `--translator`).

## Instalación con uv

```bash
# Requiere Python >=3.12 y uv 0.9+
uv sync                          # deps base: pymupdf, pytesseract, pillow, deep-translator, tqdm
uv sync --extra local-opus       # + torch, transformers, sentencepiece (traducción offline opus)
uv sync --extra local-llama      # + llama-cpp-python, huggingface-hub (traducción GGUF)
uv sync --extra unlimited        # + torch vision para Unlimited-OCR (solo NVIDIA)
```

Verifica Tesseract:
```bash
tesseract --version  # 5.5.3 en C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Uso

### 1) Solo inglés (rápido, sin traducción) — recomendado para primer run

```bash
uv run main.py --pdf "REANIMACION NEONATAL CAPI 1-4.pdf" --output_dir ./outputs --dpi 200 --no-translate
```

### 2) Inglés + español (Google, demo 3 páginas)

```bash
uv run main.py --pdf "REANIMACION NEONATAL CAPI 1-4.pdf" --output_dir ./outputs_demo --pages 3 --dpi 200
```

### 3) Full 92 páginas (EN ya generado, ES con fallback)

```bash
# Google (puede rate-limitar en docs grandes — mantiene original si falla)
uv run main.py --pdf "REANIMACION NEONATAL CAPI 1-4.pdf" --output_dir ./outputs --dpi 200 --translator google

# Offline opus (CPU, sin internet, ~5-15 min en 8GB)
uv run python translate_opus.py
# o integrado:
uv run main.py --pdf "REANIMACION NEONATAL CAPI 1-4.pdf" --output_dir ./outputs --translator opus

# Offline llama.cpp (GGUF local)
# Descarga manual si no hay auto-download:
#   https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF -> qwen2.5-0.5b-instruct-q4_k_m.gguf
#   mueve a ./models/
uv run main.py --pdf "REANIMACION NEONATAL CAPI 1-4.pdf" --output_dir ./outputs --translator llama --llama-model ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### 4) Con Unlimited-OCR (solo NVIDIA CUDA)

```bash
uv sync --extra unlimited
uv run main.py --pdf "document.pdf" --output_dir ./outputs --use-unlimited --model baidu/Unlimited-OCR
# Alternativa SGLang / vLLM ver https://github.com/baidu/Unlimited-OCR#inference
```

## Salida

```
outputs/
  REANIMACION_NEONATAL_CAPI_1-4_original.md  # EN, OCR directo (172k chars, 92 págs)
  REANIMACION_NEONATAL_CAPI_1-4_en.md        # copia de original
  REANIMACION_NEONATAL_CAPI_1-4_es.md        # ES (traducido, o copia si falló Google)
  pages/page_0001.md ... page_0092.md       # por página EN
  pages_es/page_0001.md ...                 # por página ES (si --pages <=5 o via translate_*.py)
```

Formato por página: `# Página N` + texto Tesseract + `---` (simula `<|det|>` de Unlimited-OCR).

## Resultados verificados

- **Demo 3 págs:** 5349 chars EN / 6057 chars ES, traducción precisa (ej. `Foundations of Neonatal Resuscitation` -> `Fundamentos de la reanimación neonatal`).
- **Full 92 págs:** 172437 bytes, 3364 líneas, 26494 palabras, OCR 6m06s (DPI 200, PSM 1). ES completo con Google rate-limitado mantuvo EN; re-traducción con `opus`/`llama` disponible.

## Estructura

```
main.py               # pipeline principal (pdf_to_images, ocr_tesseract, translate_auto)
translate_opus.py     # batch offline opus-mt por páginas (optimizado)
translate_google.py   # batch Google por páginas con retry/backoff
pyproject.toml        # deps uv
REANIMACION NEONATAL CAPI 1-4.pdf  # fuente (no trackeada si Git LFS)
```

## Notas técnicas

- PDF escaneado sin texto (`fitz.get_text()==""`), requiere render `fitz.Matrix(dpi/72)` + OCR.
- Tesseract PSM 1 (auto) superior a PSM 6 en este PDF; DPI 150 más rápido, 200 mejor calidad, 300 introduce artefactos `a`/`m` en bullets.
- Traducción Google falla con `No translation was found` en frases cortas con `@`/`©`/`e` bullets — el código ahora mantiene original y continúa.
- Para AMD, `torch.cuda.is_available()==False`; Unlimited-OCR no inicia servidor SGLang, hace fallback CPU sin error.

## Licencia y créditos

- Basado en [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) (Apache 2.0) y [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
- Tesseract OCR 5.5, PyMuPDF 1.27, deep-translator.
