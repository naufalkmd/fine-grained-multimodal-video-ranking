"""Read-only local visual audit of generated candidates; never relabel or train."""
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import zipfile


def build_review(contracts, train_zip, evaluation, output, limit=16):
    from PIL import Image, ImageDraw, ImageOps
    manifest = json.loads((Path(contracts) / 'manifest.json').read_text(encoding='utf-8'))
    episodes = {r['episode_id']: r for r in manifest['episodes']}
    record_path = Path(evaluation) / 'evaluated_candidates.json'
    if not record_path.is_file():
        record_path = Path(evaluation) / 'generation.json'
    suite = json.loads(record_path.read_text(encoding='utf-8'))
    records = suite['records']
    # Sample each family without consulting any model predictions or scores.
    groups = {}
    for row in records:
        if row['family'] == 'append_unmatched_auxiliary':
            continue
        groups.setdefault(row['family'] + '/' + row['type'], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
    selected = []
    while len(selected) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    panels = []
    with zipfile.ZipFile(train_zip) as archive:
        for row in selected:
            members = episodes[row['episode_id']]['image_members']
            width, height, columns = 224, 148, 4
            sheet = Image.new('RGB', (width * columns, height * ((len(members)+columns-1)//columns)), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, member in enumerate(members):
                with Image.open(io.BytesIO(archive.read(member))) as raw:
                    thumb = ImageOps.contain(raw.convert('RGB'), (width, height-20))
                x, y = (i % columns)*width, (i // columns)*height
                sheet.paste(thumb, (x + (width-thumb.width)//2, y+20))
                draw.text((x+4, y+3), str(i+1), fill='black')
            buffer = io.BytesIO()
            sheet.save(buffer, format='JPEG', quality=75)
            image = base64.b64encode(buffer.getvalue()).decode()
            swap = int(hashlib.sha256(row['id'].encode()).hexdigest(), 16) % 2
            choices = [row['positive_text'], row['negative_text']]
            if swap:
                choices.reverse()
            body = ''.join(f'<p><b>{label}</b> {html.escape(text)}</p>' for label,text in zip(('A','B'), choices))
            evidence = json.dumps(row.get('evidence', {}), ensure_ascii=False, indent=2)
            panels.append(f'''<section><h3>{html.escape(row['episode_id'])} · {html.escape(row['type'])}</h3>
<img style="max-width:100%" src="data:image/jpeg;base64,{image}" alt="All original frames in order">
{body}<details><summary>생성 근거·가정 보기</summary>
<p>참 후보로 지정한 쪽: {'B' if swap else 'A'}. 실제 부재가 검증된 오답이라는 뜻은 아닙니다.</p>
<pre>{html.escape(evidence)}</pre></details></section>''')
    document = '''<!doctype html><meta charset="utf-8"><title>선지 시각 점검</title>
<style>body{font:16px sans-serif;max-width:1050px;margin:30px auto;line-height:1.6;padding:12px}section{border-top:1px solid #ccc;padding:20px 0}pre{white-space:pre-wrap}</style>
<h1>선지 시각 점검</h1><p>모든 원래 프레임을 시간순으로 표시합니다. A/B를 읽고 두 설명 모두 맞을 가능성,
실제로 보이지 않는 행동, 문법 오류를 점검하세요. 이 화면은 학습이나 라벨 변경을 수행하지 않습니다.
가려진 행동이나 프레임 사이 행동은 이 화면만으로 확정할 수 없습니다. 이미지는 이 HTML에만 들어가며 공유 결과 ZIP에는 포함하지 않습니다.</p>''' + ''.join(panels)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding='utf-8')
    return {'selected': len(selected), 'output': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('contracts', 'train-zip', 'evaluation', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('limit must be positive')
    print(build_review(args.contracts, args.train_zip, args.evaluation, args.output, args.limit))
