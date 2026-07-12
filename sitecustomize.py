import importlib.abc
import importlib.resources.abc

# Hotfix for Python 3.14 compatibility with older MLflow releases
importlib.abc.Traversable = importlib.resources.abc.Traversable
print("[Python 3.14 Hotfix] Patched importlib.abc.Traversable")
