import sys
from pathlib import Path

# Localiza a raiz do projeto.
ROOT = Path(__file__).resolve().parents[1]

# Localiza a pasta scripts.
SCRIPTS_DIR = ROOT / "scripts"

# Adiciona os diretórios ao caminho de importação.
for directory in (ROOT, SCRIPTS_DIR):
    directory_str = str(directory)

    if directory_str not in sys.path:
        sys.path.insert(0, directory_str)