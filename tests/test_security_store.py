import tempfile
import unittest
from pathlib import Path

from backend.security import hash_password, redact, verify_password
from backend.store import Store


class SecurityStoreTests(unittest.TestCase):
    def test_password_hash_is_not_reversible(self):
        encoded = hash_password("correct horse battery")
        self.assertTrue(verify_password("correct horse battery", encoded))
        self.assertFalse(verify_password("wrong password", encoded))
        self.assertNotIn("correct horse", encoded)

    def test_redaction(self):
        value = redact({"path": "/home/alice/models/a.gguf", "authorization": "Bearer abc", "prompt": "secret"}, remove_prompts=True)
        self.assertEqual(value["prompt"], "[prompt hidden]")
        self.assertEqual(value["authorization"], "[redacted]")

    def test_store_seeds_defaults_and_roundtrips_job(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "bench.db")
            job_id = store.create_job({"models": [{"id": "demo"}], "benchmark": {"concurrencies": [1]}})
            job = store.job(job_id)
            self.assertEqual(job["config"]["models"][0]["id"], "demo")
            self.assertEqual(len(store.presets()), 3)


if __name__ == "__main__":
    unittest.main()
