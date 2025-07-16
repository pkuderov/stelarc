import torch


class LambdaReturn:
    """
    Class for TD(lambda) returns calculation.
    To compute GAE, you have to subtract V[t], i.e. in vectorised form
        GAE = lambda_ret - V

    NB: calculating lambda returns is generally more simple and effective, since
    it can be used for both actor (GAE) and critic (TD(lambda) target)
    """
    def __init__(self, gamma: float, lambda_: float = 0.95):
        self.lambda_ = lambda_
        self.gamma = gamma

    # noinspection PyPep8Naming
    def __call__(self, V, r, term, trunc, G, t):
        gamma, lambda_ = self.gamma, self.lambda_
        # NB1: _tn subscript everywhere means _{t+1} (aka t next)
        # NB2: initially, XXX[t] is effectively XXX[-1], which means
        # we grab estimates from the last timestep (full backup via V)
        G_tn = V[t]

        # NB3: the whole function is meant to compute lambda returns
        # and store them to the passed G variable
        while t > 0:
            V_tn = V[t]
            t -= 1

            backup = (1.0 - lambda_) * V_tn + lambda_ * G_tn

            # should carefully treat the end of the episode, i.e.
            # t_last -> t = 0 (next episode) transition
            G[t] = torch.where(
                term[t],
                # a) terminated: just 0
                0.,
                # non-terminal:
                torch.where(
                    trunc[t],
                    # b) truncated: backup onto V[s]
                    V[t],
                    # c) otherwise: regular r + gamma * backup
                    r[t] + gamma * backup
                )
            )
            G_tn = G[t]
