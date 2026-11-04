"""Load optional institutional helpers kept outside the distributed package."""
from functools import lru_cache
import importlib.util
import os
from pathlib import Path


@lru_cache(maxsize=1)
def load_local_archive():
    """Load an explicitly configured file or a source-checkout local override."""
    configured = os.environ.get('BRUCE_LOCAL_ARCHIVE')
    path = (Path(configured).expanduser() if configured else
            Path(__file__).resolve().parents[3] / '.bruce-local' / 'nasa_archive.py')
    if not path.is_file():
        raise RuntimeError(
            'This institutional helper is not included in BRUCE. '
            'Configure BRUCE_LOCAL_ARCHIVE with your local implementation file.'
        )
    spec = importlib.util.spec_from_file_location('_bruce_local_nasa_archive', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load the configured local archive implementation')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
