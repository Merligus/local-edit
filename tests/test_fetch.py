"""Tests for verified, resumable downloads, against a local HTTP server.

Run:  python3 tests/test_fetch.py

Resume is the part that needs a test it cannot get from the real catalog. The
largest single file here is 19.8 GB, and a connection dropped at 90% is half an
hour thrown away — but the failure mode of getting resume *wrong* is worse than
not having it: appending to a `.part` after a server ignored the `Range` header
produces a file of exactly the right length made of two overlapping copies,
which passes a size check and then fails inside ggml.

So this runs a real HTTP server on loopback: one that honours Range, and one
that quietly ignores it.
"""

import hashlib
import http.server
import socket
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from _harness import check, run                                    # noqa: E402
from local_edit.engine import fetch                                # noqa: E402

PAYLOAD = bytes(range(256)) * 4096          # 1 MiB, and position-revealing
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class Handler(http.server.BaseHTTPRequestHandler):
    honour_range = True

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range")
        if rng and self.honour_range:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range",
                             f"bytes {start}-{len(PAYLOAD)-1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def serve(honour_range=True):
    cls = type("H", (Handler,), {"honour_range": honour_range})
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = http.server.HTTPServer(("127.0.0.1", port), cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}/f.bin"


def test_plain_download():
    print("\na whole file, verified by size and checksum")
    server, url = serve()
    try:
        with TemporaryDirectory() as d:
            dest = Path(d) / "f.bin"
            fetch._download(url, dest, expect_bytes=len(PAYLOAD))
            check(dest.read_bytes() == PAYLOAD, "the bytes arrive intact")
            check(not dest.with_suffix(".bin.part").exists(),
                  "and the .part is gone")

            dest2 = Path(d) / "g.bin"
            fetch._download(url, dest2, sha256=DIGEST)
            check(dest2.read_bytes() == PAYLOAD, "a checksummed download works")
    finally:
        server.shutdown()


def test_wrong_size_is_refused():
    print("\na file of the wrong length never lands")
    server, url = serve()
    try:
        with TemporaryDirectory() as d:
            dest = Path(d) / "f.bin"
            try:
                fetch._download(url, dest, expect_bytes=len(PAYLOAD) + 1)
                check(False, "a size mismatch raises")
            except fetch.FetchError as e:
                check("expected" in str(e), f"a size mismatch raises: {e}")
            check(not dest.exists(),
                  "and nothing is installed — a truncated weight file that "
                  "looks present is worse than one that is missing")
    finally:
        server.shutdown()


def test_wrong_checksum_is_refused():
    print("\nso does a file with the wrong contents")
    server, url = serve()
    try:
        with TemporaryDirectory() as d:
            dest = Path(d) / "f.bin"
            try:
                fetch._download(url, dest, sha256="00" * 32)
                check(False, "a checksum mismatch raises")
            except fetch.FetchError as e:
                check("checksum" in str(e), f"a checksum mismatch raises: {e}")
            check(not dest.exists(), "and nothing is installed")
    finally:
        server.shutdown()


def test_resume_continues():
    print("\na half-finished download resumes where it stopped")
    server, url = serve(honour_range=True)
    try:
        with TemporaryDirectory() as d:
            dest = Path(d) / "f.bin"
            part = dest.with_name("f.bin.part")
            part.write_bytes(PAYLOAD[:400_000])          # an interrupted fetch
            fetch._download(url, dest, expect_bytes=len(PAYLOAD))
            check(dest.read_bytes() == PAYLOAD,
                  "the finished file is byte-identical to the whole payload")
    finally:
        server.shutdown()


def test_resume_falls_back_when_the_server_will_not():
    print("\na server that ignores Range does not corrupt the file")
    # The dangerous case. The server answers 200 and sends everything from byte
    # zero; appending would give a file of the wrong length made of two
    # overlapping copies. Starting over is the only safe response.
    server, url = serve(honour_range=False)
    try:
        with TemporaryDirectory() as d:
            dest = Path(d) / "f.bin"
            dest.with_name("f.bin.part").write_bytes(PAYLOAD[:400_000])
            fetch._download(url, dest, expect_bytes=len(PAYLOAD))
            check(dest.read_bytes() == PAYLOAD,
                  "the file is correct, not 1.4 MiB of overlapping halves")
    finally:
        server.shutdown()


def test_cancel_keeps_the_part():
    print("\ncancelling keeps the .part so the next attempt can resume")
    server, url = serve()
    try:
        with TemporaryDirectory() as d:
            dest = Path(d) / "f.bin"
            try:
                fetch._download(url, dest, expect_bytes=len(PAYLOAD),
                                is_cancelled=lambda: True)
                check(False, "cancelling raises Cancelled")
            except fetch.Cancelled:
                check(True, "cancelling raises Cancelled")
            check(not dest.exists(), "nothing is installed")
            check(dest.with_name("f.bin.part").exists(),
                  "but the partial file survives — throwing away 19 GB because "
                  "someone pressed Cancel would be its own bug")
    finally:
        server.shutdown()


def test_progress_is_continuous_across_files():
    print("\nprogress counts one bar across a multi-file recipe")
    server, url = serve()
    try:
        with TemporaryDirectory() as d:
            seen = []
            fetch._download(url, Path(d) / "f.bin", expect_bytes=len(PAYLOAD),
                            progress=lambda done, total, _l: seen.append(
                                (done, total)),
                            base=5_000_000, grand_total=9_000_000)
            check(seen and seen[0][0] > 5_000_000,
                  "reported progress starts from the running total, not zero")
            check(all(t == 9_000_000 for _, t in seen),
                  "and the denominator is the whole recipe, so a three-file "
                  "download does not restart its bar twice")
    finally:
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(run(
        test_plain_download, test_wrong_size_is_refused,
        test_wrong_checksum_is_refused, test_resume_continues,
        test_resume_falls_back_when_the_server_will_not,
        test_cancel_keeps_the_part, test_progress_is_continuous_across_files))
