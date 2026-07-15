from __future__ import annotations

import unittest

from rsr.run import (
    APIRequestError,
    CurlChatClient,
    SDKChatClient,
    _api_cache_matches,
    _request_chat_completion,
)


class APICacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {"testcase": "01_hard_ambiguous"}
        self.points = {
            2: {
                "localization_id": 2,
                "x": 435,
                "y": 340,
                "npz_label": 2,
                "dataset_object_id": 1,
            }
        }
        self.cached = {
            "testcase": "01_hard_ambiguous",
            "scene_id": 815,
            "split": 0,
            "annotation": "yellow plyer",
            "model": "gpt-4o",
            "localization_mode": "gt",
            "upload_image": {"transport_format": "JPEG"},
            "raw_response": "[2, yellow plyer]",
            "predicted_localization_id": 2,
            "predicted_npz_label": 2,
            "predicted_object_id": 1,
            "predicted_class_name": "yellow plyer",
            "point_x": 435,
            "point_y": 340,
            "status": "ok",
        }

    def matches(self, cached: dict) -> bool:
        return _api_cache_matches(
            cached,
            metadata=self.metadata,
            scene_id=815,
            split=0,
            instruction="yellow plyer",
            localization_mode="gt",
            model="gpt-4o",
            points_by_id=self.points,
        )

    def test_matching_successful_cache_is_reused(self) -> None:
        self.assertTrue(self.matches(self.cached))

    def test_changed_model_is_not_reused(self) -> None:
        cached = {**self.cached, "model": "different-model"}
        self.assertFalse(self.matches(cached))

    def test_failed_api_result_is_not_reused(self) -> None:
        cached = {**self.cached, "status": "unparsed_response"}
        self.assertFalse(self.matches(cached))

    def test_changed_localization_mapping_is_not_reused(self) -> None:
        cached = {**self.cached, "predicted_object_id": 99}
        self.assertFalse(self.matches(cached))


class APITimeoutTests(unittest.TestCase):
    def test_sdk_applies_seven_minutes_to_all_network_phases(self) -> None:
        client = SDKChatClient("test-key", "https://example.invalid/v1", 420.0)
        timeout = client.client.timeout
        self.assertEqual(timeout.connect, 420.0)
        self.assertEqual(timeout.read, 420.0)
        self.assertEqual(timeout.write, 420.0)
        self.assertEqual(timeout.pool, 420.0)

    def test_curl_keeps_requested_timeout(self) -> None:
        client = CurlChatClient("https://example.invalid/v1", 420.0)
        self.assertEqual(client.timeout_seconds, 420.0)


class APIRetryTests(unittest.TestCase):
    class FlakyClient:
        def __init__(self, failures: int):
            self.failures = failures
            self.calls = 0

        def create(self, **payload):
            self.calls += 1
            if self.calls <= self.failures:
                raise ConnectionError("peer reset")
            return {"choices": [{"message": {"content": "[1, object]"}}]}

    def test_retries_transport_failure_until_success(self) -> None:
        client = self.FlakyClient(failures=2)
        response, attempts = _request_chat_completion(
            client,
            {"model": "gpt-4o"},
            max_attempts=3,
            retry_backoff_seconds=0,
        )
        self.assertEqual(attempts, 3)
        self.assertEqual(client.calls, 3)
        self.assertEqual(response["choices"][0]["message"]["content"], "[1, object]")

    def test_raises_after_all_attempts_fail(self) -> None:
        client = self.FlakyClient(failures=3)
        with self.assertRaisesRegex(APIRequestError, "failed after 2 attempts"):
            _request_chat_completion(
                client,
                {"model": "gpt-4o"},
                max_attempts=2,
                retry_backoff_seconds=0,
            )


if __name__ == "__main__":
    unittest.main()
