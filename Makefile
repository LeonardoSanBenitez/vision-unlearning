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
	$(call exec_docker, poetry run --quiet pycodestyle --max-line-length=300 --ignore=E701,W605,E251,E252,E265,E303,E302,E305,W293,W291,E225,E227,E721,E741,W391,E117,E501,W503,W504 ./vision_unlearning)

	echo '\n\n-------\nPytest checks\n-------'
	$(call exec_docker, poetry run --quiet pytest ./tests)
	# Manual tests (requires things like connecting some hardware or doing something interactive)
	# poetry run --quiet pytest ./tests/**/manual_*.py
	# poetry run --quiet pytest --capture=no -k "test_example" ./tests/**/manual_example.py

build-pip: run-interactive-docker
	$(call exec_docker, poetry run --quiet python -m build)
	$(call exec_docker, poetry run --quiet twine check dist/*)
	$(call exec_docker, poetry run --quiet twine upload --skip-existing dist/*)

build-docs: run-interactive-docker
	$(call exec_docker, poetry run --quiet sphinx-build -b html docs/source docs/_build)
