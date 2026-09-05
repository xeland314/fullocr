from pathlib import Path
import re
from transformers import MarianMTModel, MarianTokenizer
import torch

pages_dir = Path('outputs/pages')
pages_es_dir = Path('outputs/pages_es')
pages_es_dir.mkdir(parents=True, exist_ok=True)
stem='REANIMACION_NEONATAL_CAPI_1-4'
model_name='Helsinki-NLP/opus-mt-en-es'
print('Loading opus-mt...', flush=True)
tok=MarianTokenizer.from_pretrained(model_name)
model=MarianMTModel.from_pretrained(model_name)
model.eval()
print('Model ready', flush=True)

def translate_batch(paras: list[str]) -> list[str]:
    """Traduce lista de párrafos en batch (más rápido que uno por uno)."""
    if not paras:
        return []
    # Filtrar vacíos pero mantener índices
    non_empty = [(i, p) for i, p in enumerate(paras) if p.strip()]
    if not non_empty:
        return paras
    texts = [p for _, p in non_empty]
    # Dividir textos largos >900 en sub-chunks (manejo simplificado: truncar a 900)
    # Para este doc, pocos paras >900, los traducimos individualmente con split
    # Aquí hacemos batch para los que cabe
    short_texts = []
    short_indices = []
    long_map = {}
    for idx, txt in non_empty:
        if len(txt) > 900:
            # split largo por oraciones y traducir por separado (no batch)
            sents=re.split(r'(?<=[.!?])\s+', txt)
            chunk=''; parts=[]
            for s in sents:
                if len(chunk)+len(s)+1>800:
                    if chunk:
                        b=tok([chunk], return_tensors='pt', padding=True, truncation=True, max_length=512)
                        with torch.no_grad():
                            g=model.generate(**b, max_length=512, num_beams=1, early_stopping=True)
                        parts.append(tok.batch_decode(g, skip_special_tokens=True)[0])
                    chunk=s
                else:
                    chunk=f'{chunk} {s}'.strip() if chunk else s
            if chunk:
                b=tok([chunk], return_tensors='pt', padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    g=model.generate(**b, max_length=512, num_beams=1, early_stopping=True)
                parts.append(tok.batch_decode(g, skip_special_tokens=True)[0])
            long_map[idx] = ' '.join(parts)
        else:
            short_indices.append(idx)
            short_texts.append(txt)
    # Batch para cortos (hasta 8 por batch para memoria)
    result = {idx: long_map[idx] for idx in long_map}
    batch_size = 8
    for i in range(0, len(short_texts), batch_size):
        batch_texts = short_texts[i:i+batch_size]
        batch_idxs = short_indices[i:i+batch_size]
        b=tok(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            g=model.generate(**b, max_length=512, num_beams=1, early_stopping=True)
        decoded=tok.batch_decode(g, skip_special_tokens=True)
        for idx, dec in zip(batch_idxs, decoded):
            result[idx]=dec
    # reconstruir orden original
    out = []
    for i, p in enumerate(paras):
        if not p.strip():
            out.append(p)
        elif i in result:
            out.append(result[i])
        else:
            out.append(p)
    return out

def translate_para(para):
    return translate_batch([para])[0]

pages=sorted(pages_dir.glob('page_*.md'))
combined=[]
for idx, p in enumerate(pages, 1):
    txt=p.read_text(encoding='utf-8')
    m=re.match(r'# Página (\d+)\n\n(.*)', txt, re.DOTALL)
    if m:
        num=m.group(1)
        body=m.group(2).replace('\n---\n','').strip()
    else:
        num=p.stem.split('_')[1]
        body=txt
    if not body.strip() or body=='*[Sin texto detectado]*':
        trans_body=body
    else:
        paras=body.split('\n\n')
        trans_paras=translate_batch(paras)
        trans_body='\n\n'.join(trans_paras)
    out_path=pages_es_dir / p.name
    out_path.write_text(f'# Página {num} (ES)\n\n{trans_body}\n\n---\n', encoding='utf-8')
    combined.append(f'# Página {num} (ES)\n\n{trans_body}\n\n---')
    if idx % 10 == 0 or idx == len(pages):
        print(f'Progreso {idx}/{len(pages)} traducidas', flush=True)

combined_text=f'# {stem} (Traducido al Español - opus-mt local)\n\n*Traducción automática en->es vía Helsinki-NLP/opus-mt-en-es (CPU, offline) - {len(pages)} páginas*\n\n---\n\n' + '\n\n'.join(combined)
Path(f'outputs/{stem}_es.md').write_text(combined_text, encoding='utf-8')
print(f'Combined ES written {len(combined_text)} chars via opus-mt', flush=True)
print('Done', flush=True)
