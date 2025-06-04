FROM rust:1-slim-bookworm

WORKDIR /app

ADD src src
ADD Cargo.toml .

RUN --mount=type=cache,target=/usr/local/cargo/registry \
	RUSTFLAGS="-C target-cpu=native" cargo build --release --bin pls
