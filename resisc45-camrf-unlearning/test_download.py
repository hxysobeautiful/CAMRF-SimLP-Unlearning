"""Offline downloader checks; local HTTP fixtures contain no real dataset."""
from __future__ import annotations

import contextlib
import hashlib
import http.server
import socket
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import prepare_resisc45 as preparation


@contextlib.contextmanager
def local_http(payload: bytes, mode: str = "normal"):
    state = {"requests": [], "first": True}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            request_range = self.headers.get("Range")
            state["requests"].append(request_range)
            cut = mode != "normal" and state["first"]
            state["first"] = False
            if cut:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload[: len(payload) // 2])
                self.wfile.flush()
                self.close_connection = True
                try:
                    self.connection.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                return
            if request_range and mode != "ignore_range":
                start = int(request_range.split("=", 1)[1].split("-", 1)[0])
                body = payload[start:]
                header_start = start + 1 if mode == "bad_range" else start
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {header_start}-{len(payload)-1}/{len(payload)}")
            else:
                body = payload
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/fixture.zip", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # Greater than a typical 1 MiB download chunk, so interruption leaves
        # at least one completed chunk regardless of urllib's read behaviour.
        self.payload = bytes(range(256)) * (16 * 1024)
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.destination = self.root / "fixture.zip"

    def tearDown(self):
        self.temp.cleanup()

    def download(self, url):
        return preparation.download_archive(url, self.destination, expected_sha256=self.sha, retries=1, timeout=5)

    def test_normal_download_and_verified_reuse(self):
        with local_http(self.payload) as (url, state):
            self.assertEqual(Path(self.download(url)), self.destination)
            self.assertEqual(self.destination.read_bytes(), self.payload)
            count = len(state["requests"])
            self.download(url)
            self.assertEqual(len(state["requests"]), count)

    def test_interrupted_download_resumes_with_range(self):
        with local_http(self.payload, "resume") as (url, state):
            with self.assertRaises(Exception):
                self.download(url)
            self.assertFalse(self.destination.exists())
            self.download(url)
            self.assertEqual(self.destination.read_bytes(), self.payload)
            self.assertTrue(any(value and value.startswith("bytes=") for value in state["requests"][1:]))

    def test_server_ignoring_range_restarts_without_append(self):
        with local_http(self.payload, "ignore_range") as (url, state):
            with self.assertRaises(Exception):
                self.download(url)
            self.download(url)
            self.assertEqual(self.destination.read_bytes(), self.payload)
            self.assertTrue(any(state["requests"][1:]))

    def test_wrong_content_range_does_not_append_partial(self):
        with local_http(self.payload, "bad_range") as (url, state):
            with self.assertRaises(Exception):
                self.download(url)
            partials = {p: p.read_bytes() for p in self.root.iterdir() if p.is_file()}
            self.assertTrue(partials, "Interrupted downloads should remain resumable")
            with self.assertRaises(Exception):
                self.download(url)
            self.assertFalse(self.destination.exists())
            for path, original in partials.items():
                if path.exists():
                    self.assertEqual(path.read_bytes(), original)
            self.assertTrue(any(state["requests"][1:]))

    def test_wrong_checksum_not_published(self):
        with local_http(self.payload) as (url, _):
            with self.assertRaises(Exception):
                preparation.download_archive(url, self.destination, expected_sha256="0" * 64, retries=1, timeout=5)
        self.assertFalse(self.destination.exists())


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.classes = mock.patch.object(preparation, "CLASSES", ("airport", "forest"))
        self.count = mock.patch.object(preparation, "IMAGES_PER_CLASS", 2)
        self.classes.start()
        self.count.start()

    def tearDown(self):
        self.count.stop()
        self.classes.stop()
        self.temp.cleanup()

    def archive(self, extra=None, classes=("airport", "forest")):
        path = self.root / "dataset.zip"
        with zipfile.ZipFile(path, "w") as stream:
            for category in classes:
                for index in range(2):
                    stream.writestr(f"NWPU-RESISC45/{category}/{category}_{index:03d}.jpg", b"fixture-jpeg")
            for name, content in extra or []:
                stream.writestr(name, content)
        return path

    def test_valid_extract_and_count(self):
        destination = self.root / "data" / "NWPU-RESISC45"
        result = preparation.extract_archive(self.archive(), destination)
        self.assertEqual(Path(result), destination)
        preparation.validate_dataset(destination)
        self.assertEqual(len(list(destination.rglob("*.jpg"))), 4)

    def test_missing_class_rejected(self):
        destination = self.root / "incomplete"
        (destination / "airport").mkdir(parents=True)
        for index in range(2):
            (destination / "airport" / f"airport_{index}.jpg").write_bytes(b"fixture")
        with self.assertRaises(Exception):
            preparation.validate_dataset(destination)

    def test_traversal_rejected_without_outside_write(self):
        destination = self.root / "data" / "NWPU-RESISC45"
        archive = self.archive(extra=[("NWPU-RESISC45/../../escaped.txt", b"escape")])
        with self.assertRaises(Exception):
            preparation.extract_archive(archive, destination)
        self.assertFalse((self.root / "escaped.txt").exists())
        self.assertFalse((self.root / "data" / "escaped.txt").exists())

    def test_nonempty_target_not_overwritten(self):
        destination = self.root / "data" / "NWPU-RESISC45"
        destination.mkdir(parents=True)
        sentinel = destination / "user_data.txt"
        sentinel.write_bytes(b"keep")
        with self.assertRaises(Exception):
            preparation.extract_archive(self.archive(), destination)
        self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_empty_image_rejected(self):
        destination = self.root / "data" / "NWPU-RESISC45"
        preparation.extract_archive(self.archive(), destination)
        (destination / "airport" / "airport_000.jpg").write_bytes(b"")
        with self.assertRaises(Exception):
            preparation.validate_dataset(destination)


if __name__ == "__main__":
    unittest.main(verbosity=2)
