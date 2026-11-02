from __future__ import annotations

import importlib
import sys

PACKAGES = ["tensorflow", "numpy", "pandas", "rasterio", "PIL", "matplotlib", "streamlit", "tqdm"]


def main():
    print(f"Python: {sys.version.split()[0]}")
    failed = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {package}: {version}")
        except Exception as error:
            failed.append(package)
            print(f"[MISSING/ERROR] {package}: {error}")
    try:
        import tensorflow as tf
        print(f"TensorFlow devices: {tf.config.list_physical_devices()}")
        print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    except Exception:
        pass
    if failed:
        raise SystemExit(f"Install missing packages: {', '.join(failed)}")
    print("Environment looks ready.")


if __name__ == "__main__":
    main()
