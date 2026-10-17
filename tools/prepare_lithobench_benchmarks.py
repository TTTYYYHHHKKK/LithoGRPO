"""Rasterize the official Manhattan GLP evaluation layouts at 1 nm/pixel.

Uses the positioning in upstream lithobench/evaluate.py:getTargets. These files
are evaluation targets; they are never used as fabricated optimized mask labels.
"""
from pathlib import Path
import json
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
MAPPING = dict(ICCAD2013='MetalSetTest', OpenROAD='ViaSetTest',
               StdMetal='StdMetal', StdContact='StdContactTest')


def rasterize(path, centered):
    polygons = []
    for line in path.read_text().splitlines():
        words = line.split()
        if len(words) < 7:
            continue
        if words[0] == 'RECT':
            x,y,w,h = map(int,words[3:7])
            polygons.append([(x,y),(x,y+h),(x+w,y+h),(x+w,y)])
        elif words[0] == 'PGON':
            coords = list(map(int,words[3:]))
            polygons.append(list(zip(coords[::2],coords[1::2])))
    if not polygons:
        raise ValueError(f'Empty GLP layout {path}')
    if centered:
        xs = [x for p in polygons for x,y in p]
        ys = [y for p in polygons for x,y in p]
        dx = (2048-(max(xs)-min(xs)))//2-min(xs)
        dy = (2048-(max(ys)-min(ys)))//2-min(ys)
    else:
        dx = dy = 384
    image = Image.new('L',(2048,2048),0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon([(x+dx,y+dy) for x,y in polygon],fill=255)
    return image


def main():
    result = {}
    for upstream, name in MAPPING.items():
        source = PROJECT/'data'/'benchmark'/upstream
        target = PROJECT/'data'/'evaluation'/name/'target'
        target.mkdir(parents=True,exist_ok=True)
        files = sorted(source.glob('*.glp'))
        for path in files:
            dest = target/(path.stem+'.png')
            if not dest.exists():
                rasterize(path,upstream in ('ICCAD2013','StdMetal')).save(dest)
        result[name] = dict(targets=len(files),path=str(target),source=str(source),pixel_nm=1,
                            renderer='Pillow integer Manhattan polygon fill; upstream placement')
    (PROJECT/'data'/'benchmark_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
