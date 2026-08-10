"""
pipeline.py

Concurrent streaming pipeline harness (Phase 1, methodology 4.1-4.2).

Three threads, following the producer-consumer structure of Ethiraj et al.
[2025]:
    - KWS thread: runs continuously on incoming audio frames. On
      detection, writes a timestamped event (plus buffered audio) to a
      thread-safe queue.
    - ASR thread: consumes detection events from the queue and
      transcribes the associated audio.
    - Measurement: wall-clock timestamps are recorded at every
      inter-thread handoff point and logged for post-hoc latency
      analysis, per methodology 4.1's evaluation convention.

IMPORTANT - what this file does NOT yet do:
    The ASR stage here is a STUB. Phase 3 (methodology 4.5) specifies
    Emformer as the real streaming ASR back-end; that hasn't been
    implemented yet, since Phase 1 only requires the harness itself to
    exist and be provably correct (threads start/stop cleanly, events
    flow through the queue in order, timestamps are captured correctly).
    Swapping the stub for a real ASR model is a Phase 3 task.

    Similarly, the KWS thread here wraps whatever trained model is
    passed in (e.g. the CRNN from models.py) but runs it once per fixed-
    length chunk rather than as a true sliding-window streaming
    detector - good enough to validate the harness's timing/threading
    correctness now; refining detection granularity is a Phase 2/3
    concern tied to the b latency-control sweep.
"""

import time
import queue
import threading
from dataclasses import dataclass


@dataclass
class DetectionEvent:
    detection_time: float
    audio_chunk: object
    chunk_id: int


@dataclass
class TimingRecord:
    chunk_id: int
    kws_start: float = None
    kws_detect: float = None
    asr_start: float = None
    asr_end: float = None

    def kws_latency_ms(self):
        if self.kws_start is None or self.kws_detect is None:
            return None
        return (self.kws_detect - self.kws_start) * 1000

    def end_to_end_latency_ms(self):
        if self.kws_start is None or self.asr_end is None:
            return None
        return (self.asr_end - self.kws_start) * 1000


class MeasurementLog:
    def __init__(self):
        self._records = {}
        self._lock = threading.Lock()

    def start_chunk(self, chunk_id, field_name, timestamp=None):
        timestamp = timestamp or time.time()
        with self._lock:
            if chunk_id not in self._records:
                self._records[chunk_id] = TimingRecord(chunk_id=chunk_id)
            setattr(self._records[chunk_id], field_name, timestamp)

    def get_all(self):
        with self._lock:
            return list(self._records.values())


class KWSWorker(threading.Thread):
    def __init__(self, model, input_queue, output_queue, measurement_log,
                 detect_fn, stop_event):
        super().__init__(daemon=True)
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.log = measurement_log
        self.detect_fn = detect_fn
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                chunk_id, audio_chunk = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self.log.start_chunk(chunk_id, "kws_start")
            is_keyword = self.detect_fn(self.model, audio_chunk)

            if is_keyword:
                self.log.start_chunk(chunk_id, "kws_detect")
                event = DetectionEvent(
                    detection_time=time.time(),
                    audio_chunk=audio_chunk,
                    chunk_id=chunk_id,
                )
                self.output_queue.put(event)

            self.input_queue.task_done()


class ASRWorker(threading.Thread):
    def __init__(self, input_queue, measurement_log, transcribe_fn, stop_event):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.log = measurement_log
        self.transcribe_fn = transcribe_fn
        self.stop_event = stop_event
        self.transcriptions = []

    def run(self):
        while not self.stop_event.is_set():
            try:
                event = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self.log.start_chunk(event.chunk_id, "asr_start")
            transcript = self.transcribe_fn(event.audio_chunk)
            self.log.start_chunk(event.chunk_id, "asr_end")
            self.transcriptions.append((event.chunk_id, transcript))

            self.input_queue.task_done()


class StreamingPipeline:
    def __init__(self, model, detect_fn, transcribe_fn):
        self.audio_queue = queue.Queue()
        self.detection_queue = queue.Queue()
        self.measurement_log = MeasurementLog()
        self.stop_event = threading.Event()

        self.kws_worker = KWSWorker(
            model=model,
            input_queue=self.audio_queue,
            output_queue=self.detection_queue,
            measurement_log=self.measurement_log,
            detect_fn=detect_fn,
            stop_event=self.stop_event,
        )
        self.asr_worker = ASRWorker(
            input_queue=self.detection_queue,
            measurement_log=self.measurement_log,
            transcribe_fn=transcribe_fn,
            stop_event=self.stop_event,
        )

    def start(self):
        self.kws_worker.start()
        self.asr_worker.start()

    def submit_audio(self, chunk_id, audio_chunk):
        self.audio_queue.put((chunk_id, audio_chunk))

    def shutdown(self, timeout=5.0):
        self.audio_queue.join()
        self.detection_queue.join()
        self.stop_event.set()
        self.kws_worker.join(timeout=timeout)
        self.asr_worker.join(timeout=timeout)

    def get_timing_report(self):
        return self.measurement_log.get_all()


if __name__ == "__main__":
    import random

    def fake_detect_fn(model, chunk):
        time.sleep(random.uniform(0.01, 0.03))
        return True

    def fake_transcribe_fn(chunk):
        time.sleep(random.uniform(0.02, 0.05))
        return f"transcript for chunk derived from {chunk}"

    pipeline = StreamingPipeline(
        model=None,
        detect_fn=fake_detect_fn,
        transcribe_fn=fake_transcribe_fn,
    )
    pipeline.start()

    n_chunks = 10
    for i in range(n_chunks):
        pipeline.submit_audio(chunk_id=i, audio_chunk=f"chunk_{i}")

    pipeline.shutdown()

    report = pipeline.get_timing_report()
    report.sort(key=lambda r: r.chunk_id)

    print(f"Processed {len(report)} chunks (expected {n_chunks})")
    assert len(report) == n_chunks, "Not all chunks were processed"

    for r in report:
        kws_ms = r.kws_latency_ms()
        e2e_ms = r.end_to_end_latency_ms()
        assert kws_ms is not None, f"chunk {r.chunk_id} missing KWS timing"
        assert e2e_ms is not None, f"chunk {r.chunk_id} missing end-to-end timing"
        print(f"chunk {r.chunk_id}: KWS latency {kws_ms:.1f}ms | "
              f"end-to-end {e2e_ms:.1f}ms")

    avg_e2e = sum(r.end_to_end_latency_ms() for r in report) / len(report)
    print(f"\nAverage end-to-end latency: {avg_e2e:.1f}ms")
    print("Pipeline harness self-test PASSED")
