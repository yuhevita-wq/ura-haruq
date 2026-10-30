"""Retired formation entry point. Use ura.manual_formations."""
from ura.manual_formations import main

def evaluate(*args, **kwargs):
    raise RuntimeError("Legacy formations retired; use ura.manual_formations.evaluate_race with full scores, lineup and pre-start odds")

if __name__ == '__main__':
    main()
