# Neural Language Model Implementation

Implementation of sequence models (LSTM/GRU) for the Deep Learning course assignment. Explores the effectiveness of different RNN architectures and regularization techniques using the Penn Treebank corpus.

## Architecture Overview
- Word2Vec embeddings (dim=200) for input representation
- Single-layer LSTM/GRU variants (hidden_dim=200)
- Implementation variants:
  * Base LSTM/GRU
  * LSTM/GRU with dropout (p=0.5)

## Implementation Details
- PyTorch implementation with automatic device selection
- Batch processing using pad_sequence for variable length inputs
- Adam optimizer (lr=1e-3)
- Cross-entropy loss function
- Early stopping based on validation perplexity


## Dependencies
```bash
pip install torch numpy matplotlib gensim nltk tqdm
```

## Execution
Requires PTB dataset in ./PTB/
```bash
python main.py
```
## Conclusions

After evaluating multiple model architectures, we obtained the following test perplexity scores:
- Base LSTM: 104.32
- LSTM with Dropout: 100.12
- GRU: 99.87
- GRU with Dropout: 95.20

The results demonstrate that:
1. GRU models outperformed LSTM models in this language modeling task
2. Adding dropout improved performance for both architectures
3. The GRU model with dropout achieved the best performance with a perplexity of 95.20
4. The addition of dropout led to approximately 4-5% improvement in perplexity for both architectures