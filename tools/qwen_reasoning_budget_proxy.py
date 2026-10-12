#!/usr/bin/env python3
"""Local streaming proxy that bounds vLLM reasoning without disabling it."""

from __future__ import annotations

import argparse
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


def inject_reasoning_budget(path: str, body: bytes, budget: int) -> bytes:
    if not path.rstrip("/").endswith("/chat/completions"):
        return body
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    chat_kwargs = payload.get("chat_template_kwargs")
    if isinstance(chat_kwargs, dict) and chat_kwargs.get("enable_thinking") is False:
        return body
    payload.setdefault("thinking_token_budget", budget)
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def make_handler(upstream_host: str, upstream_port: int, budget: int):
    class ReasoningBudgetHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            self._forward()

        def do_POST(self) -> None:  # noqa: N802
            self._forward()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._forward()

        def _forward(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            if body:
                body = inject_reasoning_budget(self.path, body, budget)

            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "content-length", "connection"}
            }
            headers["Host"] = f"{upstream_host}:{upstream_port}"
            if body:
                headers["Content-Length"] = str(len(body))

            connection: Optional[http.client.HTTPConnection] = None
            try:
                connection = http.client.HTTPConnection(
                    upstream_host, upstream_port, timeout=3600
                )
                connection.request(self.command, self.path, body=body or None, headers=headers)
                response = connection.getresponse()
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() not in {
                        "connection",
                        "content-length",
                        "transfer-encoding",
                    }:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (ConnectionError, OSError, http.client.HTTPException) as error:
                if not self.wfile.closed:
                    self.send_error(502, f"Upstream vLLM unavailable: {type(error).__name__}")
            finally:
                self.close_connection = True
                if connection is not None:
                    connection.close()

        def log_message(self, format: str, *args: object) -> None:
            print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")

    return ReasoningBudgetHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18082)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=18081)
    parser.add_argument("--thinking-token-budget", type=int, default=4096)
    args = parser.parse_args()
    if args.listen_host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the reasoning-budget proxy must bind to localhost")
    if min(args.listen_port, args.upstream_port, args.thinking_token_budget) <= 0:
        parser.error("ports and thinking-token-budget must be positive")
    return args


def main() -> None:
    args = parse_args()
    handler = make_handler(
        args.upstream_host, args.upstream_port, args.thinking_token_budget
    )
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), handler)
    print(
        f"Qwen reasoning proxy: http://{args.listen_host}:{args.listen_port} -> "
        f"http://{args.upstream_host}:{args.upstream_port}; "
        f"thinking_token_budget={args.thinking_token_budget}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

