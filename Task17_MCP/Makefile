.PHONY: run run-http build test fmt vet clean

run:
	go run ./app

# MCP over Streamable HTTP + статус-страница (для Task16 и браузера)
run-http:
	go run ./app -http 127.0.0.1:5199

build:
	go build -o bin/orbit ./app

test:
	go test ./...

fmt:
	go fmt ./...

vet:
	go vet ./...

clean:
	rm -rf bin

# Примеры прямого вызова инструментов без MCP-клиента:
probe-status:
	go run ./app -probe orbit_status

probe-natal:
	go run ./app -probe natal_build_chart -args '{"name":"Anna","birth_date":"1995-07-21","birth_time":"14:30","city":"Казань"}'

probe-vibe:
	go run ./app -probe vibe_today -args '{"city":"Казань"}'
