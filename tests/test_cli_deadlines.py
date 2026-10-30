"""Real loopback downloads: total deadlines must not become inactivity timeouts."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest

import tetolate_cli as cli


class DownloadDeadlineTests(unittest.TestCase):
    def exercise(self, *, stall=False, chunked=None):
        class Archive(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Transfer-Encoding" if chunked else "Content-Length", "chunked" if chunked else str(25 * 1024))
                self.end_headers()
                try:
                    if chunked:
                        self.wfile.write(b"1;extension=" if chunked == "header" else b"1\r\nx\r\n0\r\nTrailer: ")
                        self.wfile.flush()
                        for _ in range(25):
                            self.wfile.write(b"x")
                            self.wfile.flush()
                            time.sleep(0.04)
                        self.wfile.write(b"\r\nx\r\n0\r\n\r\n" if chunked == "header" else b"\r\n\r\n")
                        return
                    if stall:
                        time.sleep(0.2)
                    for index in range(25):
                        self.wfile.write(b"x" * 1024)
                        self.wfile.flush()
                        time.sleep(0.6 if stall and index == 0 else 0.04)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The client must close this connection at its deadline.

        with tempfile.TemporaryDirectory() as directory, ThreadingHTTPServer(("127.0.0.1", 0), Archive) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = cli.Client(f"http://127.0.0.1:{server.server_port}")
                output = Path(directory) / "archive.cbz"
                output.write_bytes(b"Keep this archive")
                started = time.monotonic()
                with self.assertRaises(cli.ClientError) as error:
                    client.download("/jobs/12345678/download/png", output, deadline=started + 0.3)
                elapsed = time.monotonic() - started
                self.assertEqual(error.exception.exit_code, cli.EXIT_TIMEOUT)
                self.assertLess(elapsed, 0.48 if stall else 0.8)
                self.assertEqual(output.read_bytes(), b"Keep this archive")
                self.assertEqual(list(Path(directory).glob(".tetolate-download-*")), [])
            finally:
                server.shutdown()
                thread.join(5)
                self.assertFalse(thread.is_alive())

    def test_slow_continuous_stream_obeys_total_deadline(self):
        self.exercise(stall=False)

    def test_stall_after_partial_data_uses_remaining_deadline(self):
        self.exercise(stall=True)

    def test_chunk_extension_cannot_extend_total_deadline(self):
        self.exercise(chunked="header")

    def test_chunk_trailer_cannot_extend_total_deadline(self):
        self.exercise(chunked="trailer")


if __name__ == "__main__":
    unittest.main()
