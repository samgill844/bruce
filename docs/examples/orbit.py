"""Save an orbit animation; use a GUI backend to see it interactively."""
from pathlib import Path
import numpy as np
from bruce.visual import animate_orbit

planet = dict(label='Planet b', t_zero=0.0, period=3.0,
              radius_1=0.1, k=0.1, incl=np.arccos(0.03), e=0.2, w=0.8)
# 60 frames across one complete orbit keeps the downloadable GIF small.
animate_orbit([planet], t=np.linspace(0, 3, 60), line_alpha=0.5,
              zoom_primary=False, save_gif=True,
              gif_path=str(Path(__file__).resolve().parents[1]/'_static/orbit.gif'),
              fps=15)
