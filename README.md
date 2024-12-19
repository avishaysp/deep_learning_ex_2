# Language Model with LSTM and GRU

This project implements a language model using LSTM and GRU architectures with Word2Vec embeddings on the Penn Treebank (PTB) dataset.

## Project Structure

```
.
├── main.py           # Main execution script
├── train.py         # Training logic implementation
├── word2vec.py      # Word2Vec model initialization
├── dataset.py       # Custom PyTorch dataset
├── data_handler.py  # Data loading utilities
├── lstm_base.py     # Base LSTM model
├── lstm_dropout.py  # LSTM with dropout
├── gru.py          # Base GRU model
├── gru_dropout.py  # GRU with dropout
└── consts.py       # Project constants
```

## Implementation Details

### Word Embeddings
- Uses Word2Vec from Gensim
- Embedding dimension: 200
- Window size: 5
- Minimum word count: 2
- Special `<pad>` token added for sequence padding

### Model Architectures

#### Base LSTM
```python
LSTM(input_size=200, hidden_size=200, num_layers=1) -> Linear(200, vocab_size)
```

#### LSTM with Dropout
```python
LSTM(input_size=200, hidden_size=200, num_layers=1) -> Dropout(0.5) -> Linear(200, vocab_size)
```

#### Base GRU
```python
GRU(input_size=200, hidden_size=200, num_layers=1) -> Linear(200, vocab_size)
```

#### GRU with Dropout
```python
GRU(input_size=200, hidden_size=200, num_layers=1) -> Dropout(0.5) -> Linear(200, vocab_size)
```

### Training Process
1. Data is loaded and tokenized from PTB dataset
2. Word2Vec model is trained on the training data
3. Sentences are converted to sequences of word embeddings
4. Model is trained using:
   - Batch size: 32
   - Learning rate: 0.001
   - Cross-entropy loss
   - Adam optimizer
   - Regular models: 10 epochs
   - Dropout models: 13 epochs

### Performance Metrics
- Models are evaluated using perplexity
- Training and validation perplexities are plotted for each epoch
- Automatic device selection (CUDA/MPS/CPU)

## Usage

1. Ensure PTB dataset is in a "PTB" directory with files:
   - ptb.train.txt
   - ptb.valid.txt

2. Install requirements:
```bash
pip install torch numpy matplotlib gensim nltk tqdm
```

3. Run the training:
```bash
python main.py
```

This will train all four models sequentially and display perplexity plots.

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

## Implementation Notes

- Uses PyTorch's `pad_sequence` for efficient batch processing
- Implements custom collate function for proper padding handling
- Automatic device selection for optimal performance
- Progress bars using tqdm for training monitoring
- Comprehensive plotting of training metrics