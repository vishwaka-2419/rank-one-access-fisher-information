from __future__ import annotations
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
FIG = ROOT / 'figures'
DATA = ROOT / 'data'
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


def save_figure(fig, name: str, dpi: int = 220):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG / f'{name}.png', dpi=dpi, bbox_inches='tight')
