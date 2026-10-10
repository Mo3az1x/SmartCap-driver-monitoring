from collections import deque


class EyeClosureTimer:
    def __init__(
        self,
        ear_threshold=0.20,
        perclos_window_seconds=30.0,
    ):
        self.ear_threshold = ear_threshold
        self.perclos_window_seconds = perclos_window_seconds

        self.eye_is_closed = False
        self.closure_start_time = None
        self.last_closure_duration_s = 0.0

        self.history = deque()

    def update(self, ear, timestamp_s):
        is_closed_now = ear < self.ear_threshold

        if is_closed_now and not self.eye_is_closed:
            self.eye_is_closed = True
            self.closure_start_time = timestamp_s

        elif not is_closed_now and self.eye_is_closed:
            self.eye_is_closed = False

            if self.closure_start_time is not None:
                self.last_closure_duration_s = (
                    timestamp_s - self.closure_start_time
                )

            self.closure_start_time = None

        if self.eye_is_closed and self.closure_start_time is not None:
            current_closure_duration_s = (
                timestamp_s - self.closure_start_time
            )
        else:
            current_closure_duration_s = 0.0

        self.history.append((timestamp_s, is_closed_now))

        cutoff_time = timestamp_s - self.perclos_window_seconds

        while self.history and self.history[0][0] < cutoff_time:
            self.history.popleft()

        perclos = self._calculate_perclos()

        return {
            "eye_state": "CLOSED" if self.eye_is_closed else "OPEN",
            "is_eye_closed": self.eye_is_closed,
            "closure_duration_s": current_closure_duration_s,
            "last_closure_duration_s": self.last_closure_duration_s,
            "perclos": perclos,
        }

    def _calculate_perclos(self):
        if len(self.history) < 2:
            return 0.0

        closed_time = 0.0
        total_time = 0.0

        for index in range(1, len(self.history)):
            previous_timestamp, previous_closed = self.history[index - 1]
            current_timestamp, _ = self.history[index]

            delta_time = current_timestamp - previous_timestamp

            if delta_time <= 0:
                continue

            total_time += delta_time

            if previous_closed:
                closed_time += delta_time

        if total_time <= 0:
            return 0.0

        return closed_time / total_time