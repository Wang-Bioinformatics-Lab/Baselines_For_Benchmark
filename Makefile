
jupyter-compose:
	docker-compose --compatibility build
	docker-compose -f docker-compose.yml --compatibility up

jupyter-compose-background:
	docker-compose --compatibility build
	docker-compose -f docker-compose.yml --compatibility up -d

jupyter-stop:
	docker-compose -f docker-compose.yml down

coder-compose:
	docker-compose -f docker-compose-coder.yml --compatibility build
	docker-compose -f docker-compose-coder.yml --compatibility up

coder-compose-background:
	docker-compose -f docker-compose-coder.yml --compatibility build
	docker-compose -f docker-compose-coder.yml --compatibility up -d

coder-stop:
	docker-compose -f docker-compose-coder.yml down