import time
import unittest

import server


class TrainingCancellationTests(unittest.TestCase):
    def setUp(self):
        server.nodes.clear()
        server.training_jobs.clear()
        server.training_batches.clear()
        server.training_queue.clear()
        server.nodes["worker"] = {
            "node_id": "worker",
            "allocated_memory_mb": 1024,
            "active_batches": 1,
        }

    def make_job(self, status="running", batch_status="queued", round_index=0, rounds=3):
        server.training_jobs["job"] = {
            "job_id": "job", "job_name": "test", "status": status,
            "current_round": round_index, "total_rounds": rounds,
            "finished_at": None, "round_metrics": [], "completed_batches": 0,
            "pending_batches": 1, "shards": [], "batch_size": 1,
        }
        server.training_batches["batch"] = {
            "batch_id": "batch", "job_id": "job", "node_id": "worker" if batch_status != "queued" else None,
            "status": batch_status, "round_index": round_index, "memory_mb": 1024,
            "weights_b64": None, "metrics": None,
        }
        server.training_queue.append("batch")

    def test_cancel_queued_job_is_immediate_and_idempotent(self):
        self.make_job()
        first = server.cancel_training_job("job")
        second = server.cancel_training_job("job")
        self.assertEqual(first["status"], "cancelled")
        self.assertEqual(second, {"job_id": "job", "status": "cancelled", "changed": False})
        self.assertEqual(server.training_batches["batch"]["status"], "cancelled")
        self.assertNotIn("batch", server.training_queue)

    def test_active_job_cancels_after_worker_ack(self):
        self.make_job(batch_status="assigned")
        response = server.cancel_training_job("job")
        self.assertEqual(response["status"], "cancelling")
        self.assertEqual(server.cancel_training_job("job")["changed"], False)
        ack = server.acknowledge_training_cancellation(
            "batch", server.AcknowledgeTrainingCancellationRequest(node_id="worker", batch_id="batch")
        )
        self.assertEqual(ack["job_status"], "cancelled")
        self.assertEqual(server.nodes["worker"]["active_batches"], 0)

    def test_terminal_and_unrelated_jobs_are_preserved(self):
        self.make_job(status="done", batch_status="done", rounds=1)
        server.training_jobs["other"] = {"job_id": "other", "status": "running"}
        response = server.cancel_training_job("job")
        self.assertEqual(response, {"job_id": "job", "status": "done", "changed": False})
        self.assertEqual(server.training_jobs["other"]["status"], "running")

    def test_late_result_is_ignored_and_does_not_advance_round(self):
        self.make_job(batch_status="assigned", round_index=1)
        server.cancel_training_job("job")
        result = server.submit_training_round_result(server.SubmitTrainingRoundResultRequest(
            node_id="worker", batch_id="batch", round_index=1, weights_b64="late", metrics={}
        ))
        self.assertEqual(result["status"], "ignored")
        self.assertEqual(server.training_jobs["job"]["current_round"], 1)
        self.assertEqual(server.training_jobs["job"]["status"], "cancelling")
        self.assertIsNone(server.training_batches["batch"]["weights_b64"])

    def test_cancel_between_rounds_prevents_enqueue_and_done(self):
        self.make_job(batch_status="queued", round_index=1)
        server.cancel_training_job("job")
        before = len(server.training_batches)
        server._enqueue_training_round_locked("job")
        server._finalize_training_round_locked("job")
        self.assertEqual(len(server.training_batches), before)
        self.assertEqual(server.training_jobs["job"]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
