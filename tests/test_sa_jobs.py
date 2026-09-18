import threading
import unittest

from src.core.sa import AnalysisCancelled, _check_cancel


class StatisticalAnalysisCancellationTests(unittest.TestCase):
    def test_cancel_event_interrupts_at_cooperative_checkpoint(self):
        event = threading.Event()
        _check_cancel(event)
        event.set()
        with self.assertRaises(AnalysisCancelled):
            _check_cancel(event)


if __name__ == '__main__':
    unittest.main()
