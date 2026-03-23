{
  "chunk_text": "Convolutional neural networks are specialized neural networks designed primarily for processing image data. Unlike traditional neural networks that connect every input neuron to every output neuron, CNNs use convolutional layers that apply small learnable filters across an image. These filters slide over the image and compute dot products with local pixel regions, producing feature maps that highlight patterns such as edges or textures. Because the same filter weights are reused across the image, CNNs require far fewer parameters compared to fully connected networks. This technique is called weight sharing. Early convolutional layers typically detect simple features like edges, while deeper layers combine these into more complex shapes and objects. This hierarchical feature learning makes CNNs highly effective for tasks such as image classification, object detection, and facial recognition.",
  "metadata": {
    "topic": "CNN",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "cnn_intermediate.md",
    "related_topics": ["convolution", "feature_maps", "filters"],
    "is_bonus": false
  }
}

{
  "chunk_text": "Pooling layers are used in convolutional neural networks to reduce the spatial size of feature maps while retaining the most important information. The most common pooling method is max pooling, which selects the largest value within a small region of the feature map, such as a 2x2 window. By keeping the strongest activation, max pooling helps the network focus on the most prominent features while reducing computational cost. Another approach is average pooling, which calculates the average value within the pooling window. Pooling layers also help provide translation invariance, meaning that the network can still recognize an object even if it appears in a slightly different position in the image. By gradually reducing feature map dimensions, pooling layers help CNNs learn increasingly abstract visual patterns.",
  "metadata": {
    "topic": "CNN",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "cnn_intermediate.md",
    "related_topics": ["max_pooling", "downsampling", "feature_maps"],
    "is_bonus": false
  }
}

{
  "chunk_text": "One of the earliest successful convolutional neural network architectures was LeNet, developed by Yann LeCun and colleagues in 1998. LeNet was designed for handwritten digit recognition and was widely used in banking systems to read zip codes and check amounts. The architecture consisted of multiple convolutional layers followed by pooling layers and fully connected layers for classification. Although small compared to modern networks, LeNet demonstrated that CNNs could automatically learn useful image features without manual feature engineering. This idea later inspired deeper architectures such as AlexNet, introduced in 2012, which significantly improved image classification accuracy on the ImageNet dataset and helped trigger the modern deep learning revolution.",
  "metadata": {
    "topic": "CNN",
    "difficulty": "intermediate",
    "type": "concept_explanation",
    "source": "cnn_intermediate.md",
    "related_topics": ["lenet", "alexnet", "imagenet"],
    "is_bonus": false
  }
}