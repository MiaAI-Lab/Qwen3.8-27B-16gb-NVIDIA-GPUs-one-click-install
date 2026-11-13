import sys, threading, time, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import serve_openai as so  # noqa: E402


class FakeJob:
    def __init__(self, name, steps):
        self.name, self.steps, self.done = name, steps, 0


class FakeGenerator:
    """Stands in for exllamav3's Generator: every iterate() advances every enqueued job by one
    step and returns one result per job; a job ends with an eos result after `steps` steps."""

    def __init__(self):
        self.jobs, self.cancelled, self.max_active, self.guard = [], [], 0, threading.Lock()
        self.fail_next = False

    def enqueue(self, job):
        self.jobs.append(job)
        self.max_active = max(self.max_active, len(self.jobs))

    def cancel(self, job):
        self.jobs.remove(job)
        self.cancelled.append(job)

    def num_remaining_jobs(self):
        return len(self.jobs)

    def iterate(self):
        assert self.guard.acquire(blocking=False), "iterate() called from two threads at once"
        try:
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("engine failure")
            time.sleep(0.001)
            out = []
            for j in list(self.jobs):
                j.done += 1
                eos = j.done >= j.steps
                out.append({"job": j, "text": f"{j.name}{j.done} ", "eos": eos})
                if eos:
                    self.jobs.remove(j)
            return out
        finally:
            self.guard.release()


def run(gate, gen, job, should_stop=None):
    got = []
    cancelled = gate.drive(gen, job, got.append, should_stop, lambda: None)
    return cancelled, got


class ParallelGate(unittest.TestCase):
    def test_every_request_gets_only_its_own_results(self):
        gate, gen = so._ParallelGate(4), FakeGenerator()
        jobs = [FakeJob(f"j{i}", 5 + i) for i in range(4)]
        res = {}
        ths = [threading.Thread(target=lambda j=j: res.update({j.name: run(gate, gen, j)})) for j in jobs]
        [t.start() for t in ths]
        [t.join(10) for t in ths]
        for j in jobs:
            cancelled, got = res[j.name]
            self.assertFalse(cancelled)
            self.assertEqual([r["text"] for r in got], [f"{j.name}{k} " for k in range(1, j.steps + 1)])
            self.assertTrue(got[-1]["eos"])
        self.assertEqual(gate.active, 0)
        self.assertEqual(gen.num_remaining_jobs(), 0)

    def test_at_most_n_jobs_in_the_engine(self):
        gate, gen = so._ParallelGate(2), FakeGenerator()
        ths = [threading.Thread(target=run, args=(gate, gen, FakeJob(f"j{i}", 20))) for i in range(6)]
        [t.start() for t in ths]
        [t.join(20) for t in ths]
        self.assertFalse(any(t.is_alive() for t in ths))
        self.assertEqual(gen.max_active, 2)

    def test_hangup_cancels_only_that_job(self):
        gate, gen = so._ParallelGate(3), FakeGenerator()
        stay, leave = FakeJob("stay", 30), FakeJob("leave", 1000)
        flag = threading.Event()
        res = {}
        t1 = threading.Thread(target=lambda: res.update(a=run(gate, gen, stay)))
        t2 = threading.Thread(target=lambda: res.update(b=run(gate, gen, leave, flag.is_set)))
        t1.start(); t2.start()
        time.sleep(0.05); flag.set()
        t1.join(10); t2.join(10)
        self.assertFalse(res["a"][0])
        self.assertTrue(res["b"][0])
        self.assertEqual(gen.cancelled, [leave])
        self.assertEqual(len(res["a"][1]), 30)
        self.assertEqual(gate.active, 0)

    def test_engine_error_reaches_the_caller_and_frees_the_slot(self):
        gate, gen = so._ParallelGate(1), FakeGenerator()
        gen.fail_next = True
        with self.assertRaises(RuntimeError):
            run(gate, gen, FakeJob("bad", 3))
        self.assertEqual(gate.active, 0)
        self.assertEqual(gen.num_remaining_jobs(), 0)      # the failed job was cancelled
        cancelled, got = run(gate, gen, FakeJob("ok", 3))  # the slot is usable again
        self.assertFalse(cancelled)
        self.assertEqual(len(got), 3)

    def test_busy_flag(self):
        gate = so._ParallelGate(2)
        self.assertFalse(gate.busy())


if __name__ == "__main__":
    unittest.main()
