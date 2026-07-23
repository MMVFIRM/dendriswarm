.PHONY: test proof proof-v02 proof-v03 proof-v04 proof-v041 proof-v051 proof-v06 proof-v07 proof-v08 check build clean

test:
	python -m compileall -q src tests scripts
	pytest

proof-v02:
	PYTHONPATH=src python scripts/run-proof.py

proof-v03:
	PYTHONPATH=src DENDRISWARM_PROOF_SEED=12345 python scripts/run-proof-v03.py
	python scripts/validate-proof-v03.py docs/PROOF_RUN_V03.json

proof-v04:
	PYTHONPATH=src python scripts/run-proof-v04.py
	python scripts/validate-proof-v04.py docs/PROOF_RUN_V04.json

proof-v041:
	PYTHONPATH=src python scripts/run-proof-v041.py
	python scripts/validate-proof-v041.py docs/PROOF_RUN_V041.json


proof-v051:
	PYTHONPATH=src python scripts/run-proof-v051.py
	PYTHONPATH=src python scripts/validate-proof-v051.py

proof-v06:
	PYTHONPATH=src python scripts/run-proof-v06.py
	PYTHONPATH=src python scripts/validate-proof-v06.py

proof-v07:
	PYTHONPATH=src python scripts/run-proof-v07.py
	PYTHONPATH=src python scripts/validate-proof-v07.py

proof-v08:
	PYTHONPATH=src python scripts/run-proof-v08.py
	PYTHONPATH=src python scripts/validate-proof-v08.py

proof: proof-v02 proof-v03 proof-v04 proof-v041 proof-v051 proof-v06 proof-v07 proof-v08

check: test proof

build:
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .demo-state
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
