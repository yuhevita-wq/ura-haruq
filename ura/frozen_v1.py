"""V1 fixed six-ticket purchase rule retired on 2026-10-30."""

def formation(*args, **kwargs):
    raise RuntimeError("V1 six-ticket formation retired; use ura.manual_formations")

if __name__ == '__main__':
    raise SystemExit("V1 retired. Run python -m ura.manual_formations --input races.json --out decisions.json")
