from enum import IntFlag

import torch


class RlFlags(IntFlag):
    TERMINATED = 1
    TRUNCATED = 2
    RESET = 4


class LambdaReturn:
    """
    Class for TD(lambda) returns calculation.
    To compute GAE, you have to subtract V[t], i.e. in vectorised form
        GAE = lambda_ret - V

    NB: calculating lambda returns is generally more simple and effective, since
    it can be used for both actor (GAE) and critic (TD(lambda) target)
    """
    def __init__(self, gamma: float, lambda_: float = None):
        self.lambda_ = lambda_
        self.gamma = gamma

    # noinspection PyPep8Naming
    @torch.no_grad()
    def __call__(self, V, r, flags, G, t):
        gamma, lambda_ = self.gamma, self.lambda_
        # adaptive lambda defined by the rollout length
        if lambda_ is None:
            lambda_ = 1.0 - 1.0 / t

        # NB1: _tn subscript everywhere means _{t+1} (aka t next)
        # NB2: initially, XXX[t] is effectively XXX[-1], which means
        # we grab estimates from the last timestep (full backup via V)
        G_tn = V[t]

        # NB3: the whole function is meant to compute lambda returns
        # and store them to the passed G variable
        rev_G = []
        while t > 0:
            V_tn = V[t]
            t -= 1

            term_tn = flags[t] & (RlFlags.TERMINATED | RlFlags.RESET)
            trunc_tn = flags[t] & RlFlags.TRUNCATED

            # since they are result of bitwise operations, their non-zero
            # "true" values != 1. Comparing explicitly with zero corrects it.
            term_tn = term_tn.bool()
            trunc_tn = trunc_tn.bool()

            # carefully estimate backups for pre-final step (t=T-1 -> t=T),
            # which means we have to compute V[s_{t+1}] = V[s_T]
            # taking into account termination and truncation flags
            backup = torch.where(
                term_tn,
                # a) terminated or reset: just 0
                #   For reset we will have G[T] = r_T + 0 = 0, since r = 0
                #   on episode reset
                0.,
                # non-terminal:
                torch.where(
                    trunc_tn,
                    # b) truncated: backup onto V[s]
                    V_tn,
                    # c) otherwise: regular backup
                    (1.0 - lambda_) * V_tn + lambda_ * G_tn
                )
            )
            G_t = r[t] + gamma * backup
            rev_G.append(G_t)

            # Because all loss calculations still require explicit handling,
            # handling it here is unnecessary/excessive. I handle it here
            # by merging this logic with termination handling. Since envs
            # return r = 0 on episode reset, we will have backup = 0 by
            # our explicit handling (see branch a), and r = 0 from env ==>
            # G[T] = 0 in the end as we wanted.

            # Note that even removing RESET from term_tn do not break anything
            # except G[T]. G[t=0...T-1] are still correct, since it is handled
            # by term/trunc flags. And G[T] should be excluded from learning
            # in either way, so we actually don't care what is stored there.

            G_tn = G_t

        # put resulting values in correct order
        G.extend(rev_G[::-1])

    # noinspection PyPep8Naming
    @torch.no_grad()
    def old_incorrect(self, V, r, term, trunc, G, t):
        gamma, lambda_ = self.gamma, self.lambda_
        # adaptive lambda defined by the rollout length
        if lambda_ is None:
            lambda_ = 1.0 - 1.0 / t

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
