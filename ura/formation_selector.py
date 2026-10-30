"""Old selector retired. Model loading remains available for historical audits."""
from ura.manual_formations import (load_probability_model as load_fixed,
    FIXED_MODEL_SHA as FIXED_HARUQ_SHA256, FIXED_URA_BLOB, main)

def choose(*args, **kwargs):
    raise RuntimeError("Old six-type catalog retired; use ura.manual_formations.evaluate_race")

if __name__ == '__main__':
    main()
