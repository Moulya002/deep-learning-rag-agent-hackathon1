{
  "chunk_text": "Artificial neural networks are computational models inspired by the structure of the human brain. They consist of layers of interconnected nodes called neurons that process information using weighted connections. A typical neural network contains an input layer, one or more hidden layers, and an output layer. The input layer receives raw data, while hidden layers perform feature transformations through weighted sums and nonlinear activation functions. The output layer produces the final prediction or classification. During training, the network adjusts the weights of these connections to minimize prediction errors. This learning process allows neural networks to identify patterns in complex datasets such as images, speech signals, or financial data. Artificial neural networks form the foundation of modern deep learning systems and are widely used in applications such as recommendation systems, medical diagnosis, and natural language processing.",
  "metadata": {
    "topic": "ANN",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "ann_intermediate.md",
    "related_topics": ["hidden_layers", "weights", "activation_functions"],
    "is_bonus": false
  }
}

{
  "chunk_text": "Backpropagation is the key algorithm used to train artificial neural networks. After a neural network produces an output prediction, the system calculates a loss value that measures how far the prediction is from the correct answer. Backpropagation works by propagating this error backward through the network to compute gradients for each weight. Using the chain rule from calculus, the algorithm determines how much each weight contributed to the error. These gradients are then used by optimization algorithms such as stochastic gradient descent to update the weights in a direction that reduces the error. Over many training iterations, this process gradually improves the network's ability to make accurate predictions. The effectiveness of backpropagation was demonstrated in the landmark 1986 research paper by Rumelhart, Hinton, and Williams, which showed how multi-layer neural networks could be efficiently trained.",
  "metadata": {
    "topic": "ANN",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "ann_intermediate.md",
    "related_topics": ["gradient_descent", "loss_function", "chain_rule"],
    "is_bonus": false
  }
}

{
  "chunk_text": "Activation functions play a critical role in neural networks because they introduce non-linearity into the model. Without activation functions, a neural network with multiple layers would behave like a simple linear model and would not be able to capture complex patterns in data. Common activation functions include the sigmoid function, hyperbolic tangent (tanh), and Rectified Linear Unit (ReLU). The sigmoid function maps values between 0 and 1, making it useful for probability predictions. The tanh function maps values between -1 and 1, often providing better centered outputs. ReLU has become the most widely used activation function in modern deep learning because it is computationally efficient and helps reduce the vanishing gradient problem. Choosing the right activation function can significantly influence how effectively a neural network learns patterns from data.",
  "metadata": {
    "topic": "ANN",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "ann_intermediate.md",
    "related_topics": ["relu", "sigmoid", "vanishing_gradient"],
    "is_bonus": false
  }
}