"""
fullocr - Pipeline OCR + Traducción basado en baidu/Unlimited-OCR

Origen: https://github.com/baidu/Unlimited-OCR
Objetivo: Obtener versión original (inglés) del PDF y versión traducida al español.

Diseño:
  - Si hay GPU NVIDIA + CUDA y el usuario pasa --use-unlimited, se usa el modelo
    baidu/Unlimited-OCR vía transformers / SGLang (ver Unlimited-OCR/infer.py).
  - Si no hay GPU (caso actual: AMD Radeon Graphics, 8GB RAM), fallback CPU:
    PyMuPDF (render) + Tesseract OCR (o RapidOCR) + deep-translator (Google).

Salida:
  outputs/<stem>_original.md  -> inglés (OCR directo, markdown por página)
  outputs/<stem>_en.md        -> copia del original (para compatibilidad)
  outputs/<stem>_es.md        -> español traducido
  outputs/pages/page_*.md     -> por página (original)
  outputs/pages_es/page_*.md  -> por página traducido (opcional)

Uso con uv:
  uv sync
  uv run main.py --pdf "REANIMACION NEONATAL CAPI 1-4.pdf" --output_dir ./outputs
  uv run main.py --pdf "file.pdf" --output_dir ./outputs --pages 5  # demo rápido
  uv run main.py --pdf "file.pdf" --output_dir ./outputs --no-translate
  uv run main.py --pdf "file.pdf" --output_dir ./outputs --use-unlimited --model baidu/Unlimited-OCR  # requiere CUDA

Referencias:
  - Unlimited-OCR README: assets, infer.py, prompt '<image>document parsing.'
  - PyMuPDF: pdf -> image (dpi 200-300)
  - Tesseract 5.5: --oem 1 --psm 1 (mejor para este PDF)
"""

import argparse
import os
import re
import sys
import time
import tempfile
import shutil
from pathlib import Path

import fitz  # pymupdf
from PIL import Image
import pytesseract
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Utilidades PDF -> Imágenes (igual que en Unlimited-OCR/infer.py)
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int = 200, tmp_dir: str | None = None) -> list[str]:
    """Convierte PDF a lista de imágenes PNG (una por página)."""
    doc = fitz.open(pdf_path)
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    else:
        os.makedirs(tmp_dir, exist_ok=True)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i+1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def get_pdf_info(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    n = len(doc)
    rect = doc[0].rect if n > 0 else None
    doc.close()
    return {"pages": n, "rect": str(rect) if rect else "N/A"}


# ---------------------------------------------------------------------------
# OCR - Tesseract (fallback CPU) + wrappers
# ---------------------------------------------------------------------------

def ocr_image_tesseract(image_path: str, lang: str = "eng", psm: int = 1, oem: int = 1) -> str:
    """OCR de una imagen con Tesseract. Devuelve texto limpio."""
    config = f"--oem {oem} --psm {psm}"
    try:
        img = Image.open(image_path)
        # Asegurar RGB
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        return clean_ocr_text(text)
    except Exception as e:
        print(f"[WARN] Tesseract failed on {image_path}: {e}", file=sys.stderr)
        return ""


def ocr_image_rapidocr(image_path: str) -> str:
    """Fallback opcional con RapidOCR (si está instalado)."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        result, _ = ocr(image_path)
        if not result:
            return ""
        # ordenar por y, luego x (top-to-bottom, left-to-right)
        result_sorted = sorted(result, key=lambda x: (x[0][0][1], x[0][0][0]))
        lines = [r[1] for r in result_sorted]
        # heuristic spacing fix: unir con espacio y luego limpiar
        text = "\n".join(lines)
        # RapidOCR pega palabras sin espacio cuando el bbox está junto; insertamos espacio si minúscula+mayúscula
        # pero no lo hacemos agresivo; dejamos tal cual y se limpia luego.
        return clean_ocr_text(text)
    except Exception as e:
        print(f"[WARN] RapidOCR not available: {e}", file=sys.stderr)
        return ""


def clean_ocr_text(text: str) -> str:
    """Limpieza mínima del output de OCR: normaliza saltos, quita artefactos."""
    # Normalizar line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Quitar líneas con solo símbolos raros (artefactos del borde)
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # descarta líneas muy cortas con solo símbolos
        if re.fullmatch(r"[^\w]{1,4}", stripped):
            continue
        # quita caracteres de control raros pero conserva acentos/bullets
        lines.append(line.rstrip())
    # colapsar 3+ saltos a 2
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # fix bullet variations: "a " o "m " al inicio (artefacto Tesseract en 300dpi con =)
    # pero mantenemos; el usuario puede revisar.
    return cleaned.strip()


def detect_tesseract_langs() -> list[str]:
    try:
        langs = pytesseract.get_languages()
        return langs
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Intento Unlimited-OCR (GPU) - inspirado en README + infer.py
# ---------------------------------------------------------------------------

def try_unlimited_ocr_infer(
    image_files: list[str],
    output_dir: str,
    model_name: str = "baidu/Unlimited-OCR",
    image_mode: str = "gundam",
    use_sglang: bool = False,
) -> bool:
    """
    Intenta usar baidu/Unlimited-OCR si hay CUDA y dependencias.
    Devuelve True si tuvo éxito, False si fallback a CPU.
    Esta función replica la lógica de Unlimited-OCR/README.md:
      model.infer(...) y model.infer_multi(...) o SGLang server.
    Requiere: torch, transformers, (opcional sglang), CUDA.
    En este equipo (AMD Radeon, 8GB) no está disponible y hará fallback.
    """
    try:
        import torch
    except ImportError:
        print("[INFO] torch no instalado -> fallback CPU (Tesseract). Para usar Unlimited-OCR: uv sync --extra unlimited")
        return False

    if not torch.cuda.is_available():
        gpu_name = "AMD Radeon (sin CUDA)" if any("AMD" in str(x) for x in [torch.cuda.get_device_name(0) if torch.cuda.device_count()>0 else ""]) else "sin GPU CUDA"
        # En Windows con AMD, torch.cuda.is_available() es False.
        print(f"[INFO] CUDA no disponible (GPU detectada: {gpu_name}) -> fallback CPU.")
        print("       Para Unlimited-OCR necesitas GPU NVIDIA + CUDA 12.9/13.0 y el wheel de SGLang.")
        return False

    # Si llegamos aquí, hay CUDA -> intentar transformers path
    try:
        from transformers import AutoModel, AutoTokenizer

        print(f"[INFO] Intentando Unlimited-OCR con transformers: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        model = model.eval().cuda()

        # Multi-page parsing (base mode para PDF)
        # image_size=1024 base, como en README multi page
        model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=image_files,
            output_path=output_dir,
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )
        print(f"[OK] Unlimited-OCR (transformers) completado -> {output_dir}")
        return True

    except Exception as e:
        print(f"[WARN] Unlimited-OCR transformers falló: {e} -> fallback CPU")
        return False


# ---------------------------------------------------------------------------
# Traducción EN -> ES
# ---------------------------------------------------------------------------

def translate_text_google(text: str, src: str = "en", tgt: str = "es", chunk_size: int = 4000) -> str:
    """
    Traduce texto con deep_translator (GoogleTranslator).
    Divide en chunks de ~4000 chars para respetar límites API.
    Si falla (sin internet), devuelve texto original con aviso.
    """
    if not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("[WARN] deep-translator no instalado, sin traducción")
        return text

    translator = GoogleTranslator(source=src, target=tgt)
    # Dividir por párrafos para preservar estructura markdown
    # Si un párrafo > chunk_size, subdividir por oraciones
    paragraphs = text.split("\n\n")
    translated_paras = []
    for para in paragraphs:
        if not para.strip():
            translated_paras.append(para)
            continue
        # si párrafo cabe en un chunk, traducir directo
        if len(para) <= chunk_size:
            try:
                translated_paras.append(translator.translate(para))
            except Exception as e:
                # Error típico: rate limit o frase no traducible -> mantener original y continuar
                print(f"[WARN] translate chunk failed: {e} -> manteniendo original")
                translated_paras.append(para)
            time.sleep(0.12)  # rate-limit más suave para evitar bloqueo Google
        else:
            # subdividir párrafo largo por oraciones
            sentences = re.split(r"(?<=[.!?])\s+", para)
            chunk = ""
            translated_chunk_parts = []
            for sent in sentences:
                if len(chunk) + len(sent) + 1 > chunk_size:
                    if chunk:
                        try:
                            translated_chunk_parts.append(translator.translate(chunk))
                        except Exception as e:
                            print(f"[WARN] translate subchunk failed: {e} -> manteniendo chunk")
                            translated_chunk_parts.append(chunk)
                        time.sleep(0.12)
                    chunk = sent
                else:
                    chunk = f"{chunk} {sent}".strip() if chunk else sent
            if chunk:
                try:
                    translated_chunk_parts.append(translator.translate(chunk))
                except Exception as e:
                    print(f"[WARN] translate last subchunk failed: {e} -> manteniendo chunk")
                    translated_chunk_parts.append(chunk)
                time.sleep(0.12)
            translated_paras.append(" ".join(translated_chunk_parts))
    return "\n\n".join(translated_paras)


# --- Local fallbacks: opus-mt y llama.cpp ---

def translate_text_opus(text: str, src: str = "en", tgt: str = "es", chunk_size: int = 4000) -> str:
    """
    Fallback offline con Helsinki-NLP/opus-mt-en-es vía transformers + torch (CPU).
    Requiere: uv sync --extra local-opus  (transformers, torch, sentencepiece)
    Si no hay torch/transformers, lanza ImportError.
    """
    if not text.strip():
        return text
    # Lazy imports
    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch
    except ImportError as e:
        raise ImportError(f"opus-mt requiere torch+transformers+sentencepiece: {e}")

    model_name = "Helsinki-NLP/opus-mt-en-es" if src == "en" and tgt == "es" else f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    print(f"[INFO] Cargando modelo local opus-mt: {model_name} (CPU, puede tardar ~30s primera vez)")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    # No movemos a CUDA (AMD no tiene CUDA); quedamos en CPU

    paragraphs = text.split("\n\n")
    out = []
    for para in paragraphs:
        if not para.strip():
            out.append(para)
            continue
        # opus-mt tiene límite ~512 tokens => dividir en chunks más pequeños (~800 chars)
        if len(para) <= 900:
            batch = tokenizer([para], return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                gen = model.generate(**batch, max_length=512, num_beams=4, early_stopping=True)
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            out.append(decoded)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            chunk = ""
            parts = []
            for sent in sentences:
                if len(chunk) + len(sent) + 1 > 800:
                    if chunk:
                        batch = tokenizer([chunk], return_tensors="pt", padding=True, truncation=True, max_length=512)
                        with torch.no_grad():
                            gen = model.generate(**batch, max_length=512, num_beams=4, early_stopping=True)
                        parts.append(tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
                    chunk = sent
                else:
                    chunk = f"{chunk} {sent}".strip() if chunk else sent
            if chunk:
                batch = tokenizer([chunk], return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    gen = model.generate(**batch, max_length=512, num_beams=4, early_stopping=True)
                parts.append(tokenizer.batch_decode(gen, skip_special_tokens=True)[0])
            out.append(" ".join(parts))
    return "\n\n".join(out)


_LLAMA_CACHE = None  # singleton para no recargar modelo en cada chunk

def _get_llama(model_path: str | None, hf_repo: str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF", n_ctx: int = 4096):
    """
    Obtiene instancia llama_cpp.Llama.
    Si model_path es None, intenta descargar GGUF pequeño vía huggingface_hub.
    Fallback: busca en ./models/*.gguf
    """
    global _LLAMA_CACHE
    if _LLAMA_CACHE is not None:
        return _LLAMA_CACHE
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(f"llama-cpp-python no instalado. Instala con: uv sync --extra local-llama ({e})")

    resolved = None
    if model_path and os.path.exists(model_path):
        resolved = model_path
    else:
        # buscar en ./models
        for cand in [model_path] if model_path else []:
            if cand and os.path.exists(cand):
                resolved = cand
                break
        if not resolved:
            models_dir = Path("./models")
            if models_dir.exists():
                ggufs = list(models_dir.glob("*.gguf"))
                if ggufs:
                    resolved = str(ggufs[0])
                    print(f"[INFO] Usando GGUF encontrado: {resolved}")
        if not resolved:
            # intentar descargar GGUF pequeño (~400MB para Qwen2.5-0.5B Q4_K_M)
            try:
                from huggingface_hub import hf_hub_download
                print(f"[INFO] Descargando GGUF desde {hf_repo} (puede tardar)...")
                # Qwen2.5-0.5B-Instruct-GGUF contiene varios quants; elegimos q4_k_m
                filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
                # algunos repos usan nombre distinto; probamos lista
                try:
                    resolved = hf_hub_download(repo_id=hf_repo, filename=filename)
                except Exception:
                    # fallback a repo TheBloke/TinyLlama
                    resolved = hf_hub_download(repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
                print(f"[OK] GGUF descargado: {resolved}")
            except Exception as e:
                raise FileNotFoundError(f"No se encontró GGUF local y la descarga falló: {e}. Descarga manual: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF y pásalo con --llama-model <path>")

    print(f"[INFO] Cargando llama.cpp modelo: {resolved} (CPU, n_ctx={n_ctx})")
    # n_gpu_layers=0 para CPU puro (AMD Radeon integrada -> Vulkan no está probado aquí)
    # Si tienes AMD con Vulkan, puedes poner n_gpu_layers=-1 y pasar --llama-vulkan
    _LLAMA_CACHE = Llama(model_path=resolved, n_ctx=n_ctx, n_threads=os.cpu_count() or 4, verbose=False)
    return _LLAMA_CACHE


def translate_text_llama_cpp(text: str, src: str = "en", tgt: str = "es", chunk_size: int = 800,
                             model_path: str | None = None, hf_repo: str | None = None) -> str:
    """
    Fallback offline con modelo GGUF vía llama-cpp-python.
    Prompt de traducción EN->ES. Útil si no hay internet.
    Requiere: uv sync --extra local-llama  y un .gguf (~400MB-1GB)
    """
    if not text.strip():
        return text
    llm = _get_llama(model_path, hf_repo=hf_repo or "Qwen/Qwen2.5-0.5B-Instruct-GGUF")

    src_name = "English" if src == "en" else src
    tgt_name = "Spanish" if tgt == "es" else tgt

    paragraphs = text.split("\n\n")
    out = []
    for para in paragraphs:
        if not para.strip():
            out.append(para)
            continue
        # Si párrafo muy largo, dividir por oraciones a chunks pequeños (llama context limitado)
        if len(para) <= chunk_size:
            chunks = [para]
        else:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            chunks, cur = [], ""
            for s in sentences:
                if len(cur) + len(s) + 1 > chunk_size:
                    if cur: chunks.append(cur)
                    cur = s
                else:
                    cur = f"{cur} {s}".strip() if cur else s
            if cur: chunks.append(cur)

        translated_chunks = []
        for chunk in chunks:
            prompt = f"Translate the following text from {src_name} to {tgt_name}. Only return the translation, no explanation.\n\nText:\n{chunk}\n\nTranslation:"
            # Usar formato chat para Qwen/TinyLlama
            # llama_cpp soporta create_chat_completion si el modelo es chat, pero usamos completion simple
            try:
                resp = llm(prompt, max_tokens=1024, temperature=0.1, stop=["\n\n", "Text:", "Translation:"])
                txt = resp["choices"][0]["text"].strip()
                # Limpieza: si el modelo repite el prompt, extraer solo después
                if "Translation:" in txt:
                    txt = txt.split("Translation:")[-1].strip()
                translated_chunks.append(txt if txt else chunk)
            except Exception as e:
                print(f"[WARN] llama_cpp chunk failed: {e}, fallback a original")
                translated_chunks.append(chunk)
        out.append(" ".join(translated_chunks))
    return "\n\n".join(out)


def translate_text_auto(text: str, src: str = "en", tgt: str = "es",
                        translator: str = "auto",
                        llama_model: str | None = None,
                        chunk_size: int = 4000) -> str:
    """
    Traductor automático con fallbacks:
      1) google (deep_translator)  - rápido, requiere internet
      2) opus-mt (local)            - offline, CPU, bueno para EN->ES
      3) llama_cpp (local GGUF)     - offline, generativo, flexible
    `translator` puede ser: auto, google, opus, llama
    """
    if translator in ("auto", "google"):
        try:
            return translate_text_google(text, src=src, tgt=tgt, chunk_size=chunk_size)
        except Exception as e:
            print(f"[WARN] GoogleTranslator falló ({e}), probando fallback local...")
            if translator == "google":
                raise
            # continuar a fallback

    if translator in ("auto", "opus"):
        try:
            return translate_text_opus(text, src=src, tgt=tgt, chunk_size=min(chunk_size, 900))
        except Exception as e:
            print(f"[WARN] opus-mt falló ({e}), probando llama_cpp...")
            if translator == "opus":
                raise

    if translator in ("auto", "llama"):
        try:
            return translate_text_llama_cpp(text, src=src, tgt=tgt, chunk_size=800, model_path=llama_model)
        except Exception as e:
            print(f"[WARN] llama_cpp falló ({e})")
            if translator == "llama":
                raise

    print("[ERROR] Todos los traductores fallaron, devolviendo texto original")
    return text


def translate_markdown_file(input_path: str, output_path: str, src: str = "en", tgt: str = "es",
                            translator: str = "auto", llama_model: str | None = None):
    text = Path(input_path).read_text(encoding="utf-8")
    translated = translate_text_auto(text, src=src, tgt=tgt, translator=translator, llama_model=llama_model)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(translated, encoding="utf-8")
    print(f"[OK] Traducido {input_path} -> {output_path} ({len(translated)} chars) via {translator}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_ocr_pipeline(
    pdf_path: str,
    output_dir: str,
    dpi: int = 200,
    lang: str = "eng",
    psm: int = 1,
    pages_limit: int | None = None,
    use_unlimited: bool = False,
    model_name: str = "baidu/Unlimited-OCR",
    do_translate: bool = True,
    src_lang: str = "en",
    tgt_lang: str = "es",
    translator: str = "auto",
    llama_model: str | None = None,
) -> dict:
    pdf_path = str(Path(pdf_path).resolve())
    output_dir = str(Path(output_dir).resolve())
    stem = Path(pdf_path).stem.replace(" ", "_")
    pages_dir = os.path.join(output_dir, "pages")
    pages_es_dir = os.path.join(output_dir, "pages_es")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)

    info = get_pdf_info(pdf_path)
    total_pages = info["pages"]
    if pages_limit is not None:
        total_pages = min(total_pages, pages_limit)
    print(f"[INFO] PDF: {pdf_path}")
    print(f"[INFO] Páginas: {info['pages']} (procesando {total_pages}), DPI={dpi}, PSM={psm}, lang={lang}")
    print(f"[INFO] Tesseract langs disponibles: {detect_tesseract_langs()[:10]}")
    print(f"[INFO] Output: {output_dir}")

    # 1) Intentar Unlimited-OCR si se pide
    if use_unlimited:
        tmp_img_dir = tempfile.mkdtemp(prefix="unlimited_ocr_imgs_")
        image_files = pdf_to_images(pdf_path, dpi=300, tmp_dir=tmp_img_dir)
        if pages_limit:
            image_files = image_files[:pages_limit]
        success = try_unlimited_ocr_infer(image_files, output_dir, model_name=model_name)
        # limpiar tmp
        # shutil.rmtree(tmp_img_dir, ignore_errors=True)  # conservar para debug
        if success:
            # Unlimited-OCR ya generó outputs; aún falta traducción
            if do_translate:
                # buscar md generado
                for md in Path(output_dir).glob("*.md"):
                    translate_markdown_file(str(md), str(Path(output_dir) / f"{md.stem}_es.md"), src=src_lang, tgt=tgt_lang, translator=translator, llama_model=llama_model)
            return {"pages": total_pages, "output_dir": output_dir, "engine": "unlimited-ocr"}

    # 2) Fallback CPU: PyMuPDF + Tesseract
    doc = fitz.open(pdf_path)
    combined_original = []
    has_error = False

    # Optimización: solo traducir por página si son pocas (demo). Para PDF grande, traducir solo el combinado.
    do_per_page_translate = do_translate and total_pages <= 5
    if do_translate and total_pages > 5:
        print(f"[INFO] PDF grande ({total_pages} págs): OCR por página será rápido, traducción solo al final (combinado).")

    print(f"[INFO] Iniciando OCR CPU (Tesseract) para {total_pages} páginas...")
    for i in tqdm(range(total_pages), desc="OCR páginas"):
        page = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        # Guardar imagen temporal
        tmp_path = os.path.join(tempfile.gettempdir(), f"fullocr_page_{i+1:04d}.png")
        pix.save(tmp_path)
        text = ocr_image_tesseract(tmp_path, lang=lang, psm=psm)
        # Fallback RapidOCR si texto muy corto y RapidOCR disponible
        if len(text.strip()) < 20:
            rapid_text = ocr_image_rapidocr(tmp_path)
            if len(rapid_text) > len(text):
                print(f"[INFO] Página {i+1}: usando RapidOCR (texto más largo)")
                text = rapid_text

        # Guardar por página
        page_md = os.path.join(pages_dir, f"page_{i+1:04d}.md")
        header = f"# Página {i+1}\n\n"
        # Añadir separador tipo Unlimited-OCR <|det|> simulado (opcional)
        page_content = header + (text if text.strip() else "*[Sin texto detectado]*") + "\n\n---\n"
        Path(page_md).write_text(page_content, encoding="utf-8")

        combined_original.append(page_content)
        # opcional traducir por página al vuelo (solo para demos pequeñas)
        if do_per_page_translate:
            os.makedirs(pages_es_dir, exist_ok=True)
            try:
                translated_page_text = translate_text_auto(text, src=src_lang, tgt=tgt_lang, translator=translator, llama_model=llama_model)
            except Exception as e:
                print(f"[WARN] traducción página {i+1} falló: {e}")
                translated_page_text = text
                has_error = True
            page_es_md = os.path.join(pages_es_dir, f"page_{i+1:04d}.md")
            Path(page_es_md).write_text(f"# Página {i+1} (ES)\n\n{translated_page_text}\n\n---\n", encoding="utf-8")

        # borrar tmp
        try:
            os.remove(tmp_path)
        except:
            pass

    doc.close()

    # Guardar combinado original (inglés)
    combined_text = f"# {Path(pdf_path).stem}\n\n*Documento OCR - {total_pages} páginas - DPI {dpi} - Engine: Tesseract (fallback CPU, inspirado en Unlimited-OCR)*\n\n---\n\n" + "\n".join(combined_original)
    original_md = os.path.join(output_dir, f"{stem}_original.md")
    en_md = os.path.join(output_dir, f"{stem}_en.md")
    Path(original_md).write_text(combined_text, encoding="utf-8")
    Path(en_md).write_text(combined_text, encoding="utf-8")
    print(f"[OK] Original (EN) guardado: {original_md} ({len(combined_text)} chars)")
    print(f"[OK] Copia EN: {en_md}")

    # Traducir combinado si procede
    if do_translate:
        print(f"[INFO] Traduciendo documento completo {src_lang}->{tgt_lang}...")
        # Si ya tenemos páginas_es (demo pequeña), reusar para evitar re-traducir
        es_chunks = []
        for i in range(total_pages):
            p = os.path.join(pages_es_dir, f"page_{i+1:04d}.md")
            if os.path.exists(p):
                es_chunks.append(Path(p).read_text(encoding="utf-8"))
        if es_chunks and len(es_chunks) == total_pages:
            combined_es = f"# {Path(pdf_path).stem} (Traducido al Español)\n\n*Traducción automática {src_lang}->{tgt_lang} vía deep-translator/Google - {total_pages} páginas*\n\n---\n\n" + "\n".join(es_chunks)
        else:
            # Traducir combinado directamente (más rápido para PDFs grandes, menos llamadas API)
            # Usar chunk_size grande y párrafos ya estructurados
            if do_per_page_translate:
                combined_es_raw = translate_text_auto(combined_text, src=src_lang, tgt=tgt_lang, translator=translator, llama_model=llama_model)
            else:
                # Para PDF grande, traducir el texto combinado sin headers de página traducidos aún
                # Extraer solo el texto de combined_original para traducir, luego reensamblar
                print(f"[INFO] Traduciendo {len(combined_text)} chars en bloques de 4000 con {translator}...")
                combined_es_raw = translate_text_auto(combined_text, src=src_lang, tgt=tgt_lang, translator=translator, llama_model=llama_model)
            combined_es = combined_es_raw
            # También generar pages_es por si se desean, a partir del combinado (opcional)
            # No crítico; el combinado ya es el output principal.
        es_md = os.path.join(output_dir, f"{stem}_es.md")
        Path(es_md).write_text(combined_es, encoding="utf-8")
        print(f"[OK] Traducido (ES) guardado: {es_md} ({len(combined_es)} chars)")
        # Si no hicimos per-page, generar también pages_es ahora dividiendo el combinado traducido
        if not do_per_page_translate and not es_chunks:
            os.makedirs(pages_es_dir, exist_ok=True)
            # Intentar dividir combined_es por "# Página" para crear pages_es individuales
            parts = re.split(r"(?=# Página \d+)", combined_es)
            for idx, part in enumerate(parts):
                if re.match(r"# Página \d+", part.strip()):
                    m = re.search(r"# Página (\d+)", part)
                    if m:
                        num = int(m.group(1))
                        page_es_file = os.path.join(pages_es_dir, f"page_{num:04d}.md")
                        Path(page_es_file).write_text(part.strip() + "\n\n---\n", encoding="utf-8")
    else:
        print("[INFO] Traducción omitida (--no-translate)")

    return {"pages": total_pages, "output_dir": output_dir, "engine": "tesseract-cpu", "original": original_md, "en": en_md, "es": os.path.join(output_dir, f"{stem}_es.md") if do_translate else None}


def main():
    parser = argparse.ArgumentParser(
        description="fullocr - OCR + traducción basado en baidu/Unlimited-OCR (fallback CPU Tesseract para AMD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pdf", required=True, help="Ruta al PDF escaneado")
    parser.add_argument("--output_dir", default="./outputs", help="Directorio de salida")
    parser.add_argument("--dpi", type=int, default=200, help="DPI para renderizado (150-300)")
    parser.add_argument("--lang", default="eng", help="Idioma Tesseract (eng, spa, etc)")
    parser.add_argument("--psm", type=int, default=1, help="Tesseract PSM (1=auto, 6=bloque)")
    parser.add_argument("--pages", type=int, default=None, help="Limitar a N páginas (demo)")
    parser.add_argument("--no-translate", action="store_true", help="No traducir al español")
    parser.add_argument("--src-lang", default="en", help="Idioma origen para traducción")
    parser.add_argument("--tgt-lang", default="es", help="Idioma destino")
    parser.add_argument("--use-unlimited", action="store_true", help="Intentar usar baidu/Unlimited-OCR (requiere NVIDIA CUDA)")
    parser.add_argument("--model", default="baidu/Unlimited-OCR", dest="model_name", help="Modelo HF para Unlimited-OCR")
    parser.add_argument("--image_mode", default="gundam", choices=("gundam", "base"), help="Modo imagen Unlimited-OCR")
    parser.add_argument("--translator", default="auto", choices=("auto", "google", "opus", "llama"), help="Motor de traducción: auto (google->opus->llama), google (requiere internet), opus (local Helsinki, CPU), llama (local GGUF via llama.cpp)")
    parser.add_argument("--llama-model", default=None, help="Ruta a .gguf para llama_cpp (ej: ./models/qwen2.5-0.5b-q4_k_m.gguf). Si no se pasa, intenta descargar TinyLlama/Qwen)")
    parser.add_argument("--llama-hf-repo", default="Qwen/Qwen2.5-0.5B-Instruct-GGUF", help="Repo HF para descargar GGUF si no existe local")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"[ERROR] PDF no encontrado: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    # Verificar Tesseract instalado
    try:
        ver = pytesseract.get_tesseract_version()
        print(f"[INFO] Tesseract {ver} detectado en: {shutil.which('tesseract')}")
    except Exception as e:
        print(f"[WARN] Tesseract no detectado: {e}. Instala https://github.com/tesseract-ocr/tesseract")

    result = run_ocr_pipeline(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        dpi=args.dpi,
        lang=args.lang,
        psm=args.psm,
        pages_limit=args.pages,
        use_unlimited=args.use_unlimited,
        model_name=args.model_name,
        do_translate=not args.no_translate,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        translator=args.translator,
        llama_model=args.llama_model,
    )
    print("\n" + "="*60)
    print("Resultado:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("="*60)
    print("Listo. Revisa ./outputs/*.md")
    print(" - *_original.md / *_en.md : versión en inglés (OCR directo)")
    print(" - *_es.md : versión traducida al español")
    print(" - pages/page_*.md : por página")


if __name__ == "__main__":
    main()
