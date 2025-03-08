FROM golang:1.23.4-alpine

WORKDIR /app

# Install build dependencies
RUN apk --no-cache add \
    ca-certificates \
    gcc \
    g++ \
    make \
    musl-dev \
    cmake \
    git \
    python3 \
    py3-pip \
    linux-headers \
    wget \
    unzip

# Create necessary directories first
RUN mkdir -p /usr/local/include /usr/local/lib

# Download and install ONNX Runtime with C API headers
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz && \
    tar -xzf onnxruntime-linux-x64-1.15.1.tgz && \
    mkdir -p onnxruntime-linux-x64-1.15.1/include && \
    mkdir -p onnxruntime-linux-x64-1.15.1/lib && \
    cp -r onnxruntime-linux-x64-1.15.1/include/* /usr/local/include/ && \
    cp -r onnxruntime-linux-x64-1.15.1/lib/* /usr/local/lib/ && \
    rm -rf onnxruntime-linux-x64-1.15.1 onnxruntime-linux-x64-1.15.1.tgz

# Set environment variables for the compiler to find the libraries
ENV CGO_CFLAGS="-I/usr/local/include"
ENV CGO_LDFLAGS="-L/usr/local/lib -lonnxruntime"
ENV LD_LIBRARY_PATH="/usr/local/lib"

# Copy go.mod and go.sum files first to leverage Docker cache
COPY go.mod go.sum* ./
RUN go mod download

# Copy the source code
COPY . .

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -o main .

# Expose port (adjust as needed)
EXPOSE 8080

# Command to run the executable
CMD ["./main"]
