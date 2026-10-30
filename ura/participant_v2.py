"""V2 is retired; CLI now runs the textbook formation policy."""
from ura.manual_formations import main, POLICY_ID

def decide(*args, **kwargs):
    raise RuntimeError("V2 rank-only six-ticket rule retired. Use ura.manual_formations.evaluate_race; full score snapshot and ordered lineup required")

if __name__ == '__main__':
    main()
