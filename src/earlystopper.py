import numpy as np


class EarlyStopper:
    def __init__(self, stopping_round, stopping_type, min_stopping_round=0, reverse=False):
        self.stopping_round = stopping_round
        self.min_stopping_round = min_stopping_round
        self.stopping_type = stopping_type
        self.last_running_loss = None
        self.count = 0
        self.reverse = reverse

    def record(self, loss):
        self.count += 1
        round_to_stop = self.count > self.min_stopping_round and self.count > self.stopping_round and \
                        self.last_running_loss is not None

        if self.stopping_type is 'moving':
            if self.last_running_loss is None:
                new_running_loss = loss
            else:
                new_running_loss = ((self.stopping_round - 1) * self.last_running_loss + loss) / self.stopping_round
            loss_to_stop = self.last_running_loss is not None and new_running_loss > self.last_running_loss
        elif self.stopping_type is 'min':
            if self.last_running_loss is None:
                new_running_loss = [loss]
            else:
                new_running_loss = [loss] + self.last_running_loss
                new_running_loss = new_running_loss[:self.stopping_round]
            loss_to_stop = np.mean(new_running_loss) > np.mean(self.last_running_loss)
        else:
            raise Exception("Stopping type is not recognized.")

        if self.reverse:
            loss_to_stop = not loss_to_stop

        if round_to_stop and loss_to_stop:
            return True
        else:
            self.last_running_loss = new_running_loss
            return False
