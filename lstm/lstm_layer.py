import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class LSTMLayer:
    def __init__(self, input_size, hidden_size, seed=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        if seed is not None:
            np.random.seed(seed)

        limit = np.sqrt(6 / (input_size + hidden_size))
        self.W_xh = np.random.uniform(-limit, limit, (input_size, 4 * hidden_size))
        self.W_hh = np.random.uniform(-limit, limit, (hidden_size, 4 * hidden_size))
        self.b = np.zeros(4 * hidden_size)

        # <<<--- CORRECTION : Initialiser le cache ici ---
        self.cache = []

    def _forward_step(self, x_t, h_prev, c_prev):
        gates = x_t @ self.W_xh + h_prev @ self.W_hh + self.b
        f_g, i_g, c_hat, o_g = np.split(gates, 4, axis=1)
        f_g, i_g, o_g = sigmoid(f_g), sigmoid(i_g), sigmoid(o_g)
        c_hat = tanh(c_hat)
        c_next = f_g * c_prev + i_g * c_hat
        h_next = o_g * tanh(c_next)
        self.cache.append((x_t, h_prev, c_prev, f_g, i_g, c_hat, o_g, c_next))
        return h_next, c_next

    def _backward_step(self, dh_next, dc_next, cache_t):
        x_t, h_prev, c_prev, f_g, i_g, c_hat, o_g, c_next = cache_t

        tanh_c_next = tanh(c_next)

        d_o_g = dh_next * tanh_c_next
        d_c_next = dh_next * o_g * (1 - tanh_c_next**2) + dc_next

        d_f_g = d_c_next * c_prev
        d_i_g = d_c_next * c_hat
        d_c_hat = d_c_next * i_g

        d_c_prev = d_c_next * f_g

        d_gates_f = d_f_g * (f_g * (1 - f_g))
        d_gates_i = d_i_g * (i_g * (1 - i_g))
        d_gates_c = d_c_hat * (1 - c_hat**2)
        d_gates_o = d_o_g * (o_g * (1 - o_g))
        d_gates = np.hstack((d_gates_f, d_gates_i, d_gates_c, d_gates_o))

        dW_xh_t = x_t.T @ d_gates
        dW_hh_t = h_prev.T @ d_gates
        db_h_t = np.sum(d_gates, axis=0)

        dx_t = d_gates @ self.W_xh.T
        dh_prev = d_gates @ self.W_hh.T

        return dx_t, dW_xh_t, dW_hh_t, db_h_t, dh_prev, d_c_prev
