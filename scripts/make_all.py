"""Regenerate every figure (and, on request, every dataset) in the package.

Default: all scripts run; the heavy scans (scripts 12--14) reuse their packaged CSV
data and only redraw their figures.  Pass --full to recompute the heavy scans too
(typically tens of minutes, depending on hardware).
"""
import runpy, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
for p in sorted(HERE.glob('[0-9][0-9]_*.py')):
    t0 = time.time()
    print(f'--- running {p.name}', flush=True)
    runpy.run_path(str(p), run_name='__main__')
    print(f'    done in {time.time()-t0:.1f}s', flush=True)
print('All figures regenerated.' + (' (full recomputation)' if '--full' in sys.argv else ' (heavy scans reused packaged data; pass --full to recompute)'))
