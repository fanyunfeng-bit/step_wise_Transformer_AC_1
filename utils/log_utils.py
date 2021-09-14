import sys


def log_values(loss, action_loss, value_loss, entropy, epoch, batch_id, step,
               tb_logger, opts):
    # Log values to screen
    print('epoch: {}, train_batch_id: {}, loss: {}, p_loss: {}, v_loss: {}, entropy: {}'.format(epoch, batch_id, loss,
                                                                                                action_loss, value_loss,
                                                                                                entropy))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('loss', loss, step)

        tb_logger.log_value('action_loss', action_loss.item(), step)
        tb_logger.log_value('value_loss', -value_loss.item(), step)

        tb_logger.log_value('entropy', entropy, step)


class Logger_terminal(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
