{
  "chunk_text": "Recurrent neural networks are designed to process sequential data where the order of inputs matters. Unlike traditional neural networks that treat each input independently, RNNs maintain a hidden state that acts as a memory of previous inputs. At each time step, the network receives a new input and updates its hidden state based on both the new input and the previous state. This structure allows the network to capture temporal dependencies in sequences such as sentences, speech signals, or stock prices. Because the same weights are reused across each time step, RNNs can process sequences of varying lengths. This ability to model sequential patterns makes recurrent neural networks useful for tasks such as language modeling, speech recognition, and time series forecasting.",
  "metadata": {
    "topic": "RNN",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "rnn_intermediate.md",
    "related_topics": ["sequence_modeling", "hidden_state"],
    "is_bonus": false
  }
}

{
  "chunk_text": "Training recurrent neural networks can be challenging due to the vanishing gradient problem. When gradients are propagated backward through many time steps during backpropagation through time, they can become extremely small. As a result, the network learns very slowly or fails to learn long-term dependencies in sequences. This problem occurs because gradients are repeatedly multiplied by small derivative values during the backward pass. Consequently, early inputs in a long sequence may have almost no influence on the learning process. Researchers developed improved architectures such as Long Short-Term Memory networks to address this limitation by allowing gradients to flow more effectively through the network.",
  "metadata": {
    "topic": "RNN",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "rnn_intermediate.md",
    "related_topics": ["bptt", "vanishing_gradient", "lstm"],
    "is_bonus": false
  }
}

{
  "chunk_text": "Long Short-Term Memory networks are a specialized type of recurrent neural network designed to capture long-range dependencies in sequential data. LSTM networks introduce a memory cell that maintains information over long time periods. The flow of information into and out of this cell is controlled by three gates: the input gate, the forget gate, and the output gate. Each gate uses a sigmoid function to decide how much information should be allowed to pass through. The forget gate determines which information should be removed from the cell state, the input gate decides which new information should be stored, and the output gate controls what information is sent to the next hidden state. This gating mechanism enables LSTMs to overcome the vanishing gradient problem and learn patterns that span long sequences.",
  "metadata": {
    "topic": "RNN",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "rnn_intermediate.md",
    "related_topics": ["lstm_gates", "sequence_learning"],
    "is_bonus": false
  }
}