# pngnw_bert



[PRETRAINED WEIGHTS LINK](https://mega.nz/folder/KQRERZwT#h23pv1xMN2zN_xqLOgytCQ)

---

## REPO STATUS: WORKS BUT NOT GOING TO BE MAINTAINED

---

Unofficial PyTorch implementation of [PnG BERT](https://arxiv.org/pdf/2103.15060.pdf) with some changes.

Dubbed "Phoneme and Grapheme and Word BERT", this model includes additional word-level embeddings on both grapheme and phoneme side of the model.

Also does include additional text-to-emoji objective using DeepMoji teacher model.

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



[1] - [LOCATION-RELATIVE ATTENTION MECHANISMS FOR ROBUST LONG-FORM
SPEECH SYNTHESIS](https://arxiv.org/pdf/1910.10288.pdf)

---

__Update 3rd March__

I've;

- added Flash Attention
- Trained Tacotron2, Prosody Prediction and Prosody-to-Mel models with PnGnW BERT
- Experimented with different Position Embedding (Learned Embedding vs Sinusoidal Embedding)

I found that - in downstream TTS tasks - fine-tuned PnGnW BERT is about on par with fine-tuning normal BERT + using DeepMoji + using G2p, while requiring **much** more VRAM and compute.

I can't recommend using this repo. The idea sounded really cool but after experimenting, it seems like the only benefit to this method is simplifying the pipeline by using a single model instead of multiple smaller models. There is no noticeable improvement in quality (which makes me really sad) and it requires 10x~ more compute.

It's still possible that this method will help a lot with accented speakers or other more challenging cases, but for normal English speakers it's just not worth it.

---
