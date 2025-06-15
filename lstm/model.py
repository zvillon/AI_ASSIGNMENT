import numpy as np
from lstm_layer import LSTMLayer


class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = np.random.randn(vocab_size, embed_dim) * 0.1

    def forward(self, idx):
        self.idx = idx
        return self.W[idx]

    def backward(self, d_out):
        dW = np.zeros_like(self.W)
        np.add.at(dW, self.idx, d_out)
        return dW


class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros(output_dim)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d_out):
        dW = self.x.T @ d_out
        db = np.sum(d_out, axis=0)
        dx = d_out @ self.W.T
        return dx, dW, db


class Encoder:
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTMLayer(embed_dim, hidden_dim)

    def forward(self, sentence_indices):
        self.lstm.cache = []
        embedded_sentence = self.embedding.forward(sentence_indices)
        h, c = np.zeros((1, self.lstm.hidden_size)), np.zeros(
            (1, self.lstm.hidden_size)
        )
        for i in range(len(embedded_sentence)):
            x_t = embedded_sentence[i].reshape(1, -1)
            h, c = self.lstm._forward_step(x_t, h, c)
        return h, c

    def backward(self, dh, dc):
        h_enc, c_enc = dh, dc
        dW_emb, dW_xh, dW_hh, db = (
            np.zeros_like(p)
            for p in [self.embedding.W, self.lstm.W_xh, self.lstm.W_hh, self.lstm.b]
        )
        for t in reversed(range(len(self.lstm.cache))):
            dx_enc, dW_xh_t, dW_hh_t, db_t, h_enc, c_enc = self.lstm._backward_step(
                h_enc, c_enc, self.lstm.cache[t]
            )
            dW_emb += self.embedding.backward(dx_enc)
            dW_xh += dW_xh_t
            dW_hh += dW_hh_t
            db += db_t
        return {"dW_emb": dW_emb, "dW_xh": dW_xh, "dW_hh": dW_hh, "db": db}


class Decoder:
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTMLayer(embed_dim, hidden_dim)
        self.fc = Linear(hidden_dim, vocab_size)

    def forward(self, token_idx, h, c):
        self.lstm.cache = []
        x = self.embedding.forward(token_idx)
        h_next, c_next = self.lstm._forward_step(x.reshape(1, -1), h, c)
        logits = self.fc.forward(h_next)
        return logits, h_next, c_next

    def backward(self, d_logits, dh_next, dc_next):
        dx_fc, dW_fc, db_fc = self.fc.backward(d_logits)
        dh_total = dx_fc + dh_next
        dx_lstm, dW_xh, dW_hh, db, dh_prev, dc_prev = self.lstm._backward_step(
            dh_total, dc_next, self.lstm.cache[-1]
        )
        dW_emb = self.embedding.backward(dx_lstm)
        grads = {
            "dW_emb": dW_emb,
            "dW_xh": dW_xh,
            "dW_hh": dW_hh,
            "db": db,
            "dW_fc": dW_fc,
            "db_fc": db_fc,
        }
        return grads, dh_prev, dc_prev


class Seq2Seq:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def save_weights(self, filename="nmt_weights.npz"):
        np.savez(
            filename,
            enc_emb_w=self.encoder.embedding.W,
            enc_lstm_wxh=self.encoder.lstm.W_xh,
            enc_lstm_whh=self.encoder.lstm.W_hh,
            enc_lstm_b=self.encoder.lstm.b,
            dec_emb_w=self.decoder.embedding.W,
            dec_lstm_wxh=self.decoder.lstm.W_xh,
            dec_lstm_whh=self.decoder.lstm.W_hh,
            dec_lstm_b=self.decoder.lstm.b,
            dec_fc_w=self.decoder.fc.W,
            dec_fc_b=self.decoder.fc.b,
        )
        print(f"\nModel weights saved to {filename}")

    def load_weights(self, filename="nmt_weights.npz"):
        try:
            data = np.load(filename, allow_pickle=True)
            self.encoder.embedding.W[:] = data["enc_emb_w"]
            self.encoder.lstm.W_xh[:] = data["enc_lstm_wxh"]
            self.encoder.lstm.W_hh[:] = data["enc_lstm_whh"]
            self.encoder.lstm.b[:] = data["enc_lstm_b"]
            self.decoder.embedding.W[:] = data["dec_emb_w"]
            self.decoder.lstm.W_xh[:] = data["dec_lstm_wxh"]
            self.decoder.lstm.W_hh[:] = data["dec_lstm_whh"]
            self.decoder.lstm.b[:] = data["dec_lstm_b"]
            self.decoder.fc.W[:] = data["dec_fc_w"]
            self.decoder.fc.b[:] = data["dec_fc_b"]
            print(f"Model weights loaded from {filename}")
        except (FileNotFoundError, KeyError) as e:
            print(
                f"Info: Could not load weights from {filename}. Starting with random weights. Reason: {e}"
            )
