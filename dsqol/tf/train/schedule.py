import numpy as np


class Schedule:

    def __call__(self, epoch: int) -> float:
        return self.call(epoch)

    def call(self, epoch: int) -> float:
        raise NotImplementedError


class ConstantSchedule(Schedule):
    def __init__(self, lr: float):
        self.lr = lr

    def call(self, epoch: int) -> float:
        return self.lr


class ExponentialDecaySchedule(Schedule):
    def __init__(self, lr: float, decay: float = 0.9):
        self.lr = lr
        self.decay = decay

    def call(self, epoch: int) -> float:
        return self.lr * self.decay ** epoch


class RestartSchedule(Schedule):
    def __init__(self, base_schedule: Schedule, period: int):
        assert period > 0
        self.base_schedule = base_schedule
        self.period = period

    def call(self, epoch: int) -> float:
        return self.base_schedule.call(epoch % self.period)
