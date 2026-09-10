# Label-Decoupled Prototype Contrast Audit

## Current B2 representation

`LabelGuidedOpcodeNet` creates the label-specific representation in the B2 branch of `src/light_label_model.py`:

```text
H: [B,T,256]
  -> attention: [B,6,T]
  -> representations Z: [B,6,256]
  -> label_scorer
  -> logits: [B,6]
```

Each `Z[:, l, :]` is the representation for one Main-6 vulnerability label. The classifier continues to consume the original unnormalized `z_l`.

## Prototype insertion point

The prototype branch reads `Z` after label-specific attention and before the end of the training step:

```text
Z [B,6,256]
  -> L2 normalization
  -> label-wise pairwise or dual-prototype loss
  -> BCE + auxiliary loss
```

The prototype branch does not change the forward classification logits or inference path. C2 stores six positive and six negative prototypes as non-trainable EMA buffers with shape `[6,256]`.

## Fair variants

- C0: existing B2 classification path with BCE only.
- C1: the same B2 path with batch-only label-decoupled supervised pairwise contrast, using `z_l` only with the same label `l`.
- C2: the same B2 path with label-wise positive and negative EMA prototypes, balanced within each label.

All variants use process01 train/valid data, seed 42, the same tokenizer/cache, embedding, BiGRU, label attention, classifier, optimizer, learning rate, epochs, early stopping and validation threshold selection. C2 prototypes are buffers and add no trainable or inference parameters.

## Invariants

The existing B0/B1/B2 code and checkpoints remain untouched. Prototype updates use train labels only and `torch.no_grad()`. Epoch 1 uses BCE only while collecting prototypes; prototype loss begins for a label only after both its positive and negative buffers initialize. Test data remains locked.
