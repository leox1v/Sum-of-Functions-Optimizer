from sfo import SFO
import torch
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer, required

class TorchSFO(Optimizer):
    """
    Sum of Functions Optimizer (SFO) for pytorch.
    The code for the SFO comes from Sohl-Dickstein et al. 'Fast large-scale optimization
    by unifying stochastic gradient and quasi-Newton methods' (https://arxiv.org/abs/1311.2115)

    Args:
        params (iterable): iterable of parameters to optimize
        dataset (tuple(Tensor, Tensor)): Tuple of Tensor of full dataset and Tensor of full labels
        batch_size (int): Batch size to use

        Example:
            optimizer = TorchSFO(model.parameters(), (X, y), batch_size=128)
            def closure(x, y_):
                f = model.forward(x)
                loss = loss_fn(f, y_)
                loss.backward()
                return loss
            optimizer.step(closure)

        """
    def __init__(self, params, dataset, batch_size):
        self.params = list(params)
        (self.data, self.target) = dataset
        self.D = len(self.target)
        self.N = int(self.D / batch_size)
        self.batch_size = batch_size

        self.optimizer = None

    def init_optimizer(self, closure):
        def f_df(newparams, data):
            x, y_ = Variable(data['x']), Variable(data['y'])
            dfdtheta = []
            for i, p in enumerate(self.params):
                if p.grad is not None:
                    p.grad.data.zero_()
                p.data = torch.from_numpy(newparams[i]).float()

            loss = closure(x, y_)

            for i, p in enumerate(self.params):
                dfdtheta.append(p.grad.data.numpy())

            loss = loss.data.numpy()
            return loss, dfdtheta

        # create the array of subfunction specific arguments
        sub_refs = []
        for i in range(self.N):
            # extract a single minibatch of training data.
            sub_refs.append({'x': self.data[i * self.batch_size:(i + 1) * self.batch_size, :, :, :],
                             'y': self.target[i * self.batch_size:(i + 1) * self.batch_size]})
        params_init = []
        for p in self.params:
            params_init.append(p.data.numpy())

        optimizer = SFO(f_df, params_init, sub_refs)
        return optimizer

    def step(self, closure):
        if self.optimizer is None:
            self.optimizer = self.init_optimizer(closure)

        thetas = self.optimizer.optimize(num_passes=1)
        return thetas



