import numpy as np
import time
import random

from data_utils import read_data, filter_pairs, normalize_string, Vocabulary
from model import Encoder, Decoder, Seq2Seq
from optimizer import SGD

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def train_iteration(model, optimizer, input_tensor, target_tensor, input_lang, output_lang):
    h, c = model.encoder.forward(input_tensor)

    decoder_token = np.array([output_lang.word2index["<SOS>"]])
    loss = 0
    d_logits_list = []
    
    for t in range(len(target_tensor)):
        logits, h, c = model.decoder.forward(decoder_token, h, c)
        probs = softmax(logits)
        loss += -np.log(probs[0, target_tensor[t]] + 1e-9)
        d_logits = probs
        d_logits[0, target_tensor[t]] -= 1
        d_logits_list.append(d_logits)
        decoder_token = np.array([target_tensor[t]])

    decoder_grads = {
        'dW_emb': np.zeros_like(model.decoder.embedding.W), 'dW_xh': np.zeros_like(model.decoder.lstm.W_xh), 
        'dW_hh': np.zeros_like(model.decoder.lstm.W_hh), 'db': np.zeros_like(model.decoder.lstm.b),
        'dW_fc': np.zeros_like(model.decoder.fc.W), 'db_fc': np.zeros_like(model.decoder.fc.b)
    }
    dh_from_decoder, dc_from_decoder = np.zeros_like(h), np.zeros_like(c)

    for t in reversed(range(len(target_tensor))):
        d_logits = d_logits_list[t]
        grads_t, dh_from_decoder, dc_from_decoder = model.decoder.backward(d_logits, dh_from_decoder, dc_from_decoder)
        for key in decoder_grads:
            decoder_grads[key] += grads_t[key]

    encoder_grads = model.encoder.backward(dh_from_decoder, dc_from_decoder)

    params = [
        model.encoder.embedding.W, model.encoder.lstm.W_xh, model.encoder.lstm.W_hh, model.encoder.lstm.b,
        model.decoder.embedding.W, model.decoder.lstm.W_xh, model.decoder.lstm.W_hh, model.decoder.lstm.b,
        model.decoder.fc.W, model.decoder.fc.b
    ]
    grads = [
        encoder_grads['dW_emb'], encoder_grads['dW_xh'], encoder_grads['dW_hh'], encoder_grads['db'],
        decoder_grads['dW_emb'], decoder_grads['dW_xh'], decoder_grads['dW_hh'], decoder_grads['db'],
        decoder_grads['dW_fc'], decoder_grads['db_fc']
    ]
    optimizer.step(zip(params, grads))
    
    return loss.item() / len(target_tensor)

def translate(model, sentence, input_lang, output_lang, max_length=10):
    unk_idx = input_lang.word2index["<UNK>"]
    input_indices = [input_lang.word2index.get(word, unk_idx) for word in sentence.split(" ")]
    h, c = model.encoder.forward(input_indices)
    decoder_token = np.array([output_lang.word2index["<SOS>"]])
    decoded_words = []
    output_vocab_rev = {i: w for w, i in output_lang.word2index.items()}
    for _ in range(max_length):
        logits, h, c = model.decoder.forward(decoder_token, h, c)
        next_token_idx = np.argmax(logits)
        if next_token_idx == output_lang.word2index["<EOS>"]:
            break
        decoded_words.append(output_vocab_rev.get(next_token_idx, "<UNK>"))
        decoder_token = np.array([next_token_idx])
    return " ".join(decoded_words)

if __name__ == "__main__":
    MAX_LENGTH = 10
    NUM_SAMPLES = 2000

    input_lang, output_lang, pairs = read_data('fra', 'eng', reverse=True)
    if pairs:
        pairs = filter_pairs(pairs, max_length=MAX_LENGTH)
        pairs = pairs[:NUM_SAMPLES]
        print(f"Using {len(pairs)} sentence pairs for training.")

        for p in pairs:
            input_lang.add_sentence(p[0])
            output_lang.add_sentence(p[1])
            p[1] += " <EOS>"
        
        embed_size = 128
        hidden_size = 256
        n_epochs = 30
        
        encoder = Encoder(len(input_lang.word2index), embed_size, hidden_size)
        decoder = Decoder(len(output_lang.word2index), embed_size, hidden_size)
        model = Seq2Seq(encoder, decoder)
        
        optimizer = SGD(lr=0.001)

        model.load_weights("nmt_weights.npz")

        print("Model created. Starting training...")
        start_time = time.time()
        
        for epoch in range(1, n_epochs + 1):
            total_loss = 0
            random.shuffle(pairs)
            for i, pair in enumerate(pairs):
                input_tensor = [input_lang.word2index.get(word, input_lang.word2index['<UNK>']) for word in pair[0].split(' ')]
                target_tensor = [output_lang.word2index.get(word, output_lang.word2index['<UNK>']) for word in pair[1].split(' ')]
                loss = train_iteration(model, optimizer, input_tensor, target_tensor, input_lang, output_lang)
                total_loss += loss
            avg_loss_epoch = total_loss / len(pairs)
            print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss_epoch:.4f}")

            if epoch % 5 == 0:
                model.save_weights("nmt_weights.npz")

        print(f"\nTraining finished in {(time.time() - start_time):.2f}s")
        print("\n--- Interactive Translation ---")
        print("Type a French sentence to translate to English, or 'quit' to exit.")
        
        while True:
            try:
                input_sentence = input("> ")
                if input_sentence.lower() == "quit":
                    break
                normalized_sentence = normalize_string(input_sentence)
                translation = translate(model, normalized_sentence, input_lang, output_lang)
                print(f"Translation: {translation}")
            except KeyboardInterrupt:
                print("\nExiting.")
                break