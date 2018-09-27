class Config(object):
  def __init__(self):
    self.learning_rate = 1e-3
    self.display_step = 50
    self.n_threads = 2
    self.regularization = 1e-3

    # Number of epochs, None is infinite
    self.n_epochs = None

