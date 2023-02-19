# pngnw_bert

---

## REPO STATUS: CODE DONE, TRAINING IN PROGRESS

---

Unofficial PyTorch implementation of [PnG BERT](https://arxiv.org/pdf/2103.15060.pdf) with some changes.

Dubbed "Phoneme and Grapheme and Word BERT", this model includes additional word-level embeddings on both grapheme and phoneme side of the model.

Also does (or will) include additional text-to-emoji objective using DeepMoji teacher model.

---

![pre_training_architecture.png](pngnwbert/pre_training_architecture.png)

Here's the modified architecture.

New stuff is

- Word Values Embeddings

- Rel Word and Rel Token Position Embeddings

- Subword Position Embeddings

- Emoji Teacher Loss

The position embeddings are configurable in the config and I will likely disable some of them once I find the best configuration for training.

---

__Update 19th Feb__

I tested 5% Trained PnGnW BERT checkpoint with Tacotron2 Decoder.

![pngnw_bert_tacotron2_alignment.png](pngnwbert/pngnw_bert_tacotron2_alignment.png)

Alignment Achieved in 300k samples, about 80% faster than the original tacotron2 text encoder [1].

I'll look into adding Flash Attention next since training is taking longer than I'd like.

---

[1] - [LOCATION-RELATIVE ATTENTION MECHANISMS FOR ROBUST LONG-FORM
SPEECH SYNTHESIS](https://arxiv.org/pdf/1910.10288.pdf)
