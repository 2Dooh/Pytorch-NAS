class AverageMeter:
    def __init__(self) -> None:
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def _update(self, val, n=1, **kwargs):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def get_value(self):
        return self.avg

class ErrorRate(AverageMeter):
    def __init__(self) -> None:
        super().__init__()

    def _update(self, real, pred, **kwargs):
        errors = pred.numel() - pred.eq(real.view_as(pred)).sum().item()
        return super()._update(errors, n=pred.numel(), **kwargs)

class Loss(AverageMeter):
    def __init__(self) -> None:
        super().__init__()

    def _update(self, loss, pred, **kwargs):
        return super()._update(loss, n=pred.numel(), **kwargs)


class AverageMeterList:
    def __init__(self, n_classes) -> None:
        self.n_classes = n_classes
        self.value = [0] * self.n_classes
        self.avg = [0] * self.n_classes
        self.sum = [0] * self.n_classes
        self.count = [0] * self.n_classes

    def reset(self):
        self.value = [0] * self.n_classes
        self.avg = [0] * self.n_classes
        self.sum = [0] * self.n_classes
        self.count = [0] * self.n_classes

    def _update(self, val, n=1, **kwargs):
        for i in range(self.n_classes):
            self.value[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    @property
    def get_value(self):
        return self.avg