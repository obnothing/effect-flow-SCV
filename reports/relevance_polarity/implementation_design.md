# Relevance–Polarity Implementation Design

## Controlled question

Does locating one label-specific evidence support before polarity discrimination improve multi-label vulnerability detection over P11's independent positive/negative token retrieval?

## Frozen components

The opcode data, split, tokenizer, 8192-token limit, embedding, one-layer BiGRU, hidden size, bidirectionality, four attention heads, scorer, loss, DoS soft targets, optimizer, learning rate, weight decay, gradient clipping, effective batch size, epoch budget, early stopping, AMP, seed, and threshold protocol remain identical to P11.

## B1: positive locator control

The positive polarity query is projected through the existing `W_Q` and produces the only token-level relevance distribution. Positive and negative polarity queries are projected by the same `W_Q` only to form feature gates over the already aggregated head features. B1 has the same trainable parameter count as P11.

## B2: dedicated relevance locator

B2 adds only `relevance_queries:[6,768]`, or 4,608 parameters. These queries produce the only token-level relevance distribution. Positive and negative queries never produce token softmax distributions; they form gates `1+tanh(W_Q q)` over the aligned head evidence.

## Shared scoring

Both variants preserve one label-wise scorer shared across polarity branches and two branch-specific biases. Final logits remain `e+ - e-`.

## Structural invariants

- Exactly one Q/K/V/O projection set exists.
- Exactly one token distribution `alpha_rel:[B,6,4,T]` exists per model forward.
- Positive and negative branches receive the same `z_rel:[B,6,4,192]`.
- B2 parameter count minus B1 parameter count must equal 4,608.
- Test data are neither loaded nor referenced.
