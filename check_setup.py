"""check_setup.py - verifies that the environment is ready. Run from the project root."""
import importlib
import sys

import config

config.setup_logging()


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    ok = True

    for pkg in ["torch", "torchvision", "ultralytics", "cv2", "numpy", "pandas",
                "sklearn", "matplotlib", "seaborn", "streamlit", "PIL", "tqdm"]:
        try:
            mod = importlib.import_module(pkg)
            print(f"  [OK]      {pkg:<12} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"  [MISSING] {pkg}")
            ok = False

    print(f"Compute device: {config.get_device()}")

    config.ensure_dirs()
    for d in config.ALL_DIRS:
        print(f"  [{'OK' if d.exists() else 'MISSING'}] {d}")

    print("\nSETUP OK" if ok else "\nSETUP INCOMPLETE: fix the MISSING items above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())