import sys
import shutil
from pathlib import Path

if len(sys.argv) < 3:
    print('Usage: prepare_dataset.py <source_path> <out_zip_path>')
    sys.exit(2)

src = Path(sys.argv[1])
out = Path(sys.argv[2])

if not src.exists():
    print('MISSING', src)
    sys.exit(3)

if src.is_dir():
    # make_archive expects base_name without extension
    base = out.with_suffix('')
    shutil.make_archive(str(base), 'zip', root_dir=str(src))
    print('ZIPPED', out.resolve())
else:
    # copy file to out
    shutil.copy2(str(src), str(out))
    print('COPIED', out.resolve())

print('SIZE:', out.stat().st_size)
