.PHONY: run build test fmt vet clean

run:
	go run ./app

build:
	go build -o bin/ai-chat ./app

test:
	go test ./...

fmt:
	go fmt ./...

vet:
	go vet ./...

clean:
	rm -rf bin
