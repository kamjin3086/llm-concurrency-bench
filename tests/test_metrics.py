import unittest

from backend.metrics import batch_metrics, percentile, sweet_spot


class MetricsTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([1, 2, 4, 8], .5), 3)

    def test_batch_keeps_failures_and_excludes_first_token(self):
        result = batch_metrics([
            {"ok": True, "started_s": 0, "first_token_s": 1, "finished_s": 3, "prompt_tokens": 5, "completion_tokens": 5, "ttft_s": 1, "stream_decode_tps": 2},
            {"ok": False, "started_s": 0, "first_token_s": None, "finished_s": 1, "prompt_tokens": 0, "completion_tokens": 0, "error": "boom"},
        ])
        self.assertEqual(result["failed_requests"], 1)
        self.assertEqual(result["completion_tokens"], 5)
        self.assertAlmostEqual(result["aggregate_decode_tps"], 2)

    def test_sweet_spot_uses_guardrails(self):
        rows = [
            {"concurrency": 1, "aggregate_decode_tps": 50, "avg_stream_decode_tps": 50, "ttft_p50_s": .1, "failed_requests": 0},
            {"concurrency": 2, "aggregate_decode_tps": 90, "avg_stream_decode_tps": 38, "ttft_p50_s": .2, "failed_requests": 0},
            {"concurrency": 4, "aggregate_decode_tps": 100, "avg_stream_decode_tps": 20, "ttft_p50_s": .4, "failed_requests": 0},
        ]
        self.assertEqual(sweet_spot(rows)["concurrency"], 2)


if __name__ == "__main__":
    unittest.main()
