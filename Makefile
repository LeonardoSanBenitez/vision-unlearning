##############################
# Global setup
.ONESHELL: # Source: https://stackoverflow.com/a/30590240
.SILENT: # https://stackoverflow.com/a/11015111


##############################
# Functions
# If a function may be executed inside `exec_cluster`, then it must be written in a single line
# Otherwise it may be "normal"
define exec_docker
	# Run a command inside a running container
	# Arguments:
	#  $(1): Command to run
	# Returns:
	#  Output of the command
	docker compose exec notebooks $(1)
endef

define is_docker_running
	# Check if the docker container is running
	# Arguments
	#  $(1): Container name
	# Returns:
	#  Exit code 0 if running, 1 if not running
	docker compose ps --filter "status=running" | grep $(1) | grep -v Exit >/dev/null
endef

##############################
# Targets for docker container
# Mainly for local development
# Only interactive mode is supported (but potentially would make sense to have batch mode too)
# TODO: push the image to dockerhub
run-interactive-docker:
	echo '\n-------------------------------------------------------------------------\n'
	echo 'Go to http://localhost:8889/tree?token=7e500f07-fe1c-44b4-b2c5-bad27bbb17f9"'
	echo '\n-------------------------------------------------------------------------\n'
	if $(call is_docker_running,vision-unlearning-notebooks); then \
		echo 'Container is already running.'
	else
		echo 'Container is not running. Starting it now...'
		docker compose down
		docker compose up -d
	fi;

clean-docker:
	docker compose down --rmi all
	make run-docker

stop-docker:
	docker compose down

##############################
# Targets for testing
# Currently all tests run inside the docker container, but that's just because of the dependencies
test: run-interactive-docker
	echo '\n\n------------------------\nMypy Check\n------------------------'
	$(call exec_docker, poetry run --quiet mypy --install-types --non-interactive > /dev/null 2>&1)  # hidden output
	$(call exec_docker, poetry run --quiet mypy --no-warn-incomplete-stub --disable-error-code import-untyped --explicit-package-bases --check-untyped-defs ./vision_unlearning)

	echo '\n\n------------------------\nPycodestyle Check\n------------------------'
	# E701: suppressed globally (multiple statements on one line)
	# W605,E251,E252: common in research/generated code (regex escapes, spaces around defaults)
	# E265,E303,E302,E305: comment style and blank-line conventions tolerated in research files
	# W293,W291: trailing whitespace in research files — cosmetic only
	# E225,E227: operator spacing in research files — cosmetic only
	# E721: type comparison style — research code uses type() directly
	# E741: ambiguous variable names (l, O, I) in research code
	# W391: blank line at end of file — cosmetic only
	# E117,E501: over-indent and long lines — cosmetic only
	# W503,W504: line break before/after binary operator (PEP8 style choice — codebase uses W503 style)
	# NOTE: this line intentionally does NOT use the exec_docker macro. `$(call ...)`
	# splits its argument on every literal comma, and the --ignore list below is a
	# comma-separated list -- so `$(call exec_docker, ...--ignore=E701,W605,...)` silently
	# truncated to `--ignore=E701` (everything after the first comma became unused $(2),
	# $(3), ... arguments to the macro). Calling docker directly sidesteps the problem.
	# --exclude matches pycodestyle.yml's u_care exclusion, PLUS reports/: fixing the
	# truncation bug above (verified this session, see PLAN-TASK-2026-07-01-TestTooling.md)
	# means this line now actually executes with the full ignore list for the first time,
	# which surfaced that reports/ (git-ignored, so invisible to CI, but present on disk
	# from prior local research sessions) has ~100 pre-existing style violations of its
	# own. Excluding it here matches `make test-lite`'s exclusion and keeps this check
	# testing the tracked repo, not whatever a developer happens to have on disk locally.
	docker compose exec notebooks poetry run --quiet pycodestyle --max-line-length=300 --ignore=E701,W605,E251,E252,E265,E303,E302,E305,W293,W291,E225,E227,E721,E741,W391,E117,E501,W503,W504 --exclude=vision_unlearning/benchmarks/u_care,vision_unlearning/benchmarks/I_care/reports ./vision_unlearning

	echo '\n\n-------\nPytest checks\n-------'
	$(call exec_docker, poetry run --quiet pytest ./tests)
	# Manual tests (requires things like connecting some hardware or doing something interactive)
	# poetry run --quiet pytest ./tests/**/manual_*.py
	# poetry run --quiet pytest --capture=no -k "test_example" ./tests/**/manual_example.py

##############################
# Lite tier: fast local development proxy for the I-CARE analysis path.
# Runs on the HOST (no Docker Desktop dependency) against a plain venv with
# requirements-test-lite.txt (no torch, no diffusers, no jax). This is a PROXY only --
# CI (GitHub Actions) is the authoritative merge gate, and `make test` (Docker, full
# dependency stack) is the full-parity proxy required before declaring a task done.
# See CONTRIBUTING.md Section 6.
# Windows venvs put executables in Scripts/, not bin/ -- $(OS) is set to "Windows_NT" by
# the environment on Windows (including Git Bash), so this picks the right layout without
# needing a separate Windows-specific target. All tools are invoked via `python -m` so we
# never depend on the entry-point wrapper scripts (identical behaviour on both layouts).
ifeq ($(OS),Windows_NT)
	VENV_LITE_PY := .venv-lite/Scripts/python
else
	VENV_LITE_PY := .venv-lite/bin/python
endif

test-lite:
	if [ ! -d .venv-lite ]; then python3 -m venv .venv-lite; fi
	$(VENV_LITE_PY) -m pip install --quiet --upgrade pip
	$(VENV_LITE_PY) -m pip install --quiet -r requirements-test-lite.txt

	echo '\n\n------------------------\nMypy Check (lite tier)\n------------------------'
	$(VENV_LITE_PY) -m mypy --no-warn-incomplete-stub --disable-error-code import-untyped --explicit-package-bases --check-untyped-defs ./vision_unlearning

	echo '\n\n------------------------\nPycodestyle Check (lite tier)\n------------------------'
	# --exclude adds reports/ on top of the CI ignore list: reports/ is git-ignored (never
	# seen by CI) but exists on disk locally with looser research-script style.
	$(VENV_LITE_PY) -m pycodestyle --max-line-length=300 --ignore=E701,W605,E251,E252,E265,E303,E302,E305,W293,W291,E225,E227,E721,E741,W391,E117,E501,W503,W504 --exclude=vision_unlearning/benchmarks/u_care,vision_unlearning/benchmarks/I_care/reports ./vision_unlearning

	echo '\n\n-------\nPytest checks (lite tier -- heavy files excluded via tests/conftest.py)\n-------'
	PYTHONPATH=. $(VENV_LITE_PY) -m pytest -m "not gpu" tests/

build-pip: run-interactive-docker
	$(call exec_docker, poetry run --quiet python -m build)
	$(call exec_docker, poetry run --quiet twine check dist/*)
	$(call exec_docker, poetry run --quiet twine upload --skip-existing dist/*)

build-docs: run-interactive-docker
	$(call exec_docker, poetry run --quiet sphinx-build -b html docs/source docs/_build)
