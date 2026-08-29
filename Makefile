.PHONY: install run test init-db

install:
	pip install -r requirements.txt

run:
	python main.py

test:
	pytest -q

init-db:
	psql "$${DATABASE_URL}" -f sql/init_db.sql
