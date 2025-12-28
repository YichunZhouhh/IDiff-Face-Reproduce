import random
from torchvision import transforms


class ProbabilisticTransform:
    def __init__(self, transform1, transform2, max_prob=0, total_warmup_epochs=2):
        self.transform1 = transform1
        self.transform2 = transform2
        self.total_warmup_epochs = total_warmup_epochs
        self.prob = 0.
        self.max_prob = max_prob
        self.current_epoch = 0

    def __call__(self, x):
        if random.random() < self.prob:
            return self.transform1(x)
        else:
            return self.transform2(x)

    def adjust_probability(self):
        # Linearly increase 'a' during warm-up
        if self.current_epoch <= self.total_warmup_epochs:
            self.prob = (self.current_epoch / self.total_warmup_epochs) * self.max_prob
        else:
            self.prob = self.max_prob

    def update_epoch(self, epoch):
        self.current_epoch = epoch
        self.adjust_probability()
        print(f"Epoch {epoch}: Updated probability to {self.prob:.4f}")
