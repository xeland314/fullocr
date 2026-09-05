from pathlib import Path
import re, time
from deep_translator import GoogleTranslator

pages_dir = Path('outputs/pages')
pages_es_dir = Path('outputs/pages_es')
pages_es_dir.mkdir(parents=True, exist_ok=True)
stem='REANIMACION_NEONATAL_CAPI_1-4'
translator = GoogleTranslator(source='en', target='es')
pages = sorted(pages_dir.glob('page_*.md'))
print(f'Traduciendo {len(pages)} páginas con GoogleTranslator (delay 0.8s por párrafo)...', flush=True)
combined=[]
failed=[]
for idx, p in enumerate(pages, 1):
    txt = p.read_text(encoding='utf-8')
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
        trans_paras=[]
        for para in paras:
            if not para.strip():
                trans_paras.append(para)
                continue
            # si párrafo >3500, dividir
            if len(para) > 3500:
                sents=re.split(r'(?<=[.!?])\s+', para)
                chunk=''; parts=[]
                for s in sents:
                    if len(chunk)+len(s)+1>3000:
                        # traducir chunk
                        for attempt in range(3):
                            try:
                                parts.append(translator.translate(chunk))
                                break
                            except Exception as e:
                                if attempt==2:
                                    print(f'  chunk page {num} failed after 3: {e}', flush=True)
                                    parts.append(chunk)
                                else:
                                    time.sleep(2*(attempt+1))
                        time.sleep(0.85)
                        chunk=s
                    else:
                        chunk=f'{chunk} {s}'.strip() if chunk else s
                if chunk:
                    for attempt in range(3):
                        try:
                            parts.append(translator.translate(chunk))
                            break
                        except Exception as e:
                            if attempt==2:
                                print(f'  chunk final page {num} failed: {e}', flush=True)
                                parts.append(chunk)
                            else:
                                time.sleep(2*(attempt+1))
                    time.sleep(0.85)
                trans_paras.append(' '.join(parts))
            else:
                for attempt in range(3):
                    try:
                        trans_paras.append(translator.translate(para))
                        break
                    except Exception as e:
                        if attempt==2:
                            print(f'  para page {num} failed: {e} -> keep original', flush=True)
                            trans_paras.append(para)
                            failed.append(num)
                        else:
                            wait=2*(attempt+1)
                            print(f'  retry page {num} attempt {attempt+1} wait {wait}s: {e}', flush=True)
                            time.sleep(wait)
                time.sleep(0.85)
        trans_body='\n\n'.join(trans_paras)
    out_path=pages_es_dir / p.name
    out_path.write_text(f'# Página {num} (ES)\n\n{trans_body}\n\n---\n', encoding='utf-8')
    combined.append(f'# Página {num} (ES)\n\n{trans_body}\n\n---')
    if idx % 10 == 0 or idx==len(pages):
        print(f'Progreso {idx}/{len(pages)}', flush=True)

combined_text=f'# {stem} (Traducido al Español)\n\n*Traducción automática en->es vía deep-translator/Google - {len(pages)} páginas*\n\n---\n\n' + '\n\n'.join(combined)
Path(f'outputs/{stem}_es.md').write_text(combined_text, encoding='utf-8')
Path(f'outputs/{stem}_es_google.md').write_text(combined_text, encoding='utf-8')
print(f'Combined ES written {len(combined_text)} chars via Google, failed pages: {failed}', flush=True)
print('Done', flush=True)
