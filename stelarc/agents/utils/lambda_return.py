import torch


class LambdaReturn:
    """Class for TD(lambda) returns calculation.
    It's NOT GAE! I just kept the naming for convenience.
    To calc GAE, you have to subtract V[t], i.e. in vectorised form
        GAE = lambda_ret - V

    NB: calculating lambda returns is more neat generally, since
    it can be used for both actor (GAE) and critic (TD(lambda) target)
    """
    def __init__(self, gamma: float, lambda_: float = 0.95):
        self.lambda_ = lambda_
        self.gamma = gamma

    # noinspection PyPep8Naming
    def __call__(self, batch):
        gamma, lambda_ = self.gamma, self.lambda_
        V, r, term, trunc = batch.v, batch.r, batch.term, batch.trunc
        # holds lambda-returns
        G = batch.lambda_ret
        t = batch.n_steps
        # _tn subscript everywhere means _{t+1} (aka t next)
        G_tn = V[t]

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

        return G
