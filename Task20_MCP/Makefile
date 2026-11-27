.PHONY: run run-http build test fmt vet clean

run:
	go run ./app

# MCP over Streamable HTTP + статус-страница (для чата Task16/17/18/19)
run-http:
	go run ./app -http 127.0.0.1:5200

build:
	go build -o bin/affirm ./app

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
	go run ./app -probe affirmation_status

probe-affirmation:
	go run ./app -probe daily_affirmation

probe-translate:
	go run ./app -probe translate_text -args '{"text":"Everything will work out just fine"}'
