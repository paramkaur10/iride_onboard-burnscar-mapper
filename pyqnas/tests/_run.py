"""Run the test_* functions of a module without pytest: python tests/test_x.py (pytest runs them too)."""
import sys
import traceback


def run(namespace):
    tests = [(n, f) for n, f in namespace.items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:                                   # noqa: BLE001
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")
    print("ALL PASS" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
