FROM golang:1.23-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -o /councilai ./cmd/server

FROM alpine:3.20
# CA certificates are required for HTTPS calls to model providers.
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /councilai /app/councilai
EXPOSE 8080
CMD ["/app/councilai"]
