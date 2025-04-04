#!/usr/bin/env python3

from sysconfig import get_paths
from pathlib import Path
import numpy as np

py_includes = get_paths()['include']
np_includes = Path(np.__path__[0]) / 'core' / 'include'

print(f'-I{py_includes} -I{np_includes}')
