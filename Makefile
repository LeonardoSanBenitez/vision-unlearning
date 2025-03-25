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
	docker-compose exec notebooks $(1)
endef

define is_docker_running
	# Check if the docker container is running
	# Arguments
	#  $(1): Container name
	# Returns:
	#  Exit code 0 if running, 1 if not running
	docker-compose ps --filter "status=running" | grep $(1) | grep -v Exit >/dev/null
endef

##############################
# Targets for docker container
# Mainly for local development
# Only interactive mode is supported (but potentially would make sense to have batch mode too)
# TODO: push the image to dockerhub
run-interactive-docker:
	echo '\n-------------------------------------------------------------------------\n'
	echo 'Go to http://localhost:8888/tree?token=7e500f07-fe1c-44b4-b2c5-bad27bbb17f9"'
	echo '\n-------------------------------------------------------------------------\n'
	if $(call is_docker_running,vision-unlearning-notebooks); then \
		echo 'Container is already running.'
	else
		echo 'Container is not running. Starting it now...'
		docker-compose down
		docker-compose up -d
	fi;

clean-docker:
	docker-compose down --rmi all
	make run-docker

stop-docker:
	docker-compose down

##############################
# Targets for testing
# Currently all tests run inside the docker container, but that's just because of the dependencies
test: run-interactive-docker
	echo '\n\n------------------------\nMypy Check\n------------------------'
	$(call exec_docker, poetry run mypy --install-types --non-interactive > /dev/null 2>&1)  # hidden output
	$(call exec_docker, poetry run mypy --no-warn-incomplete-stub --disable-error-code import-untyped --explicit-package-bases ./vision_unlearning)

	echo '\n\n------------------------\nPycodestyle Check\n------------------------'
	$(call exec_docker, poetry run pycodestyle --max-line-length=200 --ignore=E701 ./vision_unlearning)

	echo '\n\n-------\nPytest checks\n-------'
	$(call exec_docker, poetry run pytest ./tests)
	# Manual tests (requires things like connecting some hardware or doing something interactive)
	# poetry run pytest ./tests/**/manual_*.py
	# poetry run pytest --capture=no -k "test_example" ./tests/**/manual_example.py

build-pip:
	$(call exec_docker, poetry run build)
	$(call exec_docker, poetry run twine check dist/*)
	$(call exec_docker, poetry run twine upload --skip-existing dist/*)

build-docs:
	$(call exec_docker, poetry run sphinx-build -b html docs/source docs/_build)
