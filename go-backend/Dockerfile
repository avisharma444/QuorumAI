FROM golang:1.23-alpine AS builder
WORKDIR /src
COPY go.mod *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -o /councilai .
FROM alpine:3.20
COPY --from=builder /councilai /councilai
EXPOSE 8080
CMD ["/councilai"]
