# A-Novel-Approach-for-Arabic-SMS-Spam-Detection-Using-Hybrid-Deep-Learning-Techniques
A Novel Approach for Arabic SMS Spam Detection Using Hybrid Deep Learning Techniques 

Welcome to the repository for our paper titled "A Novel Approach for Arabic SMS Spam Detection Using Hybrid Deep Learning Techniques," presented at the 6th International Conference on AI in Computational Linguistics.

# Abstract
Spam detection in SMS communication is crucial for maintaining the quality of messaging services and protecting users from unwanted and potentially harmful messages. Arabic SMS spam detection poses unique challenges due to the rich morphology and complex structure of the Arabic language, which can significantly impact the performance of traditional text classification methods.

This paper presents a novel approach for Arabic SMS spam detection using a hybrid deep learning model that combines Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (Bi-LSTM) networks. The proposed model leverages the strengths of CNNs in capturing local features and patterns in the text and the capability of Bi-LSTM networks to understand long-term dependencies and contextual information. This hybrid architecture is designed to effectively handle the complexities of the Arabic language and improve the accuracy of spam detection.

# Key Features
Hybrid Deep Learning Model: Combines CNN and Bi-LSTM networks to capture both local and sequential features in Arabic SMS messages.
High Performance: Achieved an accuracy of 96.99%, precision of 97.39%, recall of 96.75%, and F1 score of 97.07%.
Comprehensive Evaluation: Includes visualizations of the confusion matrix, ROC curve, and training-validation loss graph to illustrate the model's performance.
# Methodology
The figure below illustrates the architecture of the proposed hybrid model for detecting spam in Arabic SMS messages. The model combines Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (Bi-LSTM) networks to leverage their respective strengths in capturing local patterns and long-term dependencies.

Embedding Layer: Converts tokens into dense vectors of fixed size (128 dimensions), mapping each word in the input sequence to a continuous vector space.
CNN Layer: Utilizes 64 filters with a kernel size of 5 to perform convolution operations and extract local features from the input sequences.
MaxPooling Layer: Applies max-pooling with a pool size of 4 to reduce the dimensionality of the feature maps and retain the most significant features.
Bi-LSTM Layer: Contains 64 LSTM units with dropout (0.2) and recurrent dropout (0.2), processing sequences bidirectionally to capture long-term dependencies.
Dense Layer: Outputs the final classification (ham or spam) using a sigmoid activation function.

![image](https://github.com/MohdRasol/A-Novel-Approach-for-Arabic-SMS-Spam-Detection-Using-Hybrid-Deep-Learning-Techniques/assets/59788704/affb3415-63f5-4eef-93ad-2203a072cca2)


# Dataset Preparation:

The dataset is derived from the SMS Spam Collection dataset available from the UCI Machine Learning Repository.
Translated English SMS messages into Arabic using ChatGPT-3.5 to create a high-quality Arabic dataset.
Preprocessing steps include text cleaning, tokenization, and padding.
Model Architecture:

# Embedding Layer: Converts tokens into dense vectors.
CNN Layer: Captures local patterns with multiple filters and max-pooling.
Bi-LSTM Layer: Understands long-term dependencies and contextual information with bidirectional processing.
Dense Layer: Outputs the final classification using a sigmoid activation function.
Training and Evaluation:

Model trained using the Adam optimizer and binary cross-entropy loss function.
Early stopping implemented to prevent overfitting.
Performance evaluated using accuracy, precision, recall, and F1 score metrics.
Results
The hybrid CNN-Bi-LSTM model outperformed traditional methods and standalone deep learning models in detecting Arabic SMS spam.
High accuracy and robustness demonstrated through comprehensive evaluation metrics and visualizations.
Conclusion and Future Work
The proposed hybrid model offers a robust solution for accurately classifying Arabic SMS messages. Future work includes exploring data augmentation techniques, leveraging transfer learning with pre-trained models, and investigating advanced hybrid architectures to further enhance the model's performance.

# Repository Contents
code/: Contains the implementation of the hybrid CNN-Bi-LSTM model.
data/: Includes the translated Arabic SMS dataset used for training and evaluation.
results/: Contains performance metrics and visualizations from the experiments.
docs/: Includes the full research paper and related documentation.
How to Cite
If you use this code or dataset in your research, please cite our paper:

@inproceedings{al_saidat_2024,
  title={A Novel Approach for Arabic SMS Spam Detection Using Hybrid Deep Learning Techniques},
  author={Al Saidat, Mohammed Rasol and Shalaan, Khaled and Yerima, Suleiman},
  booktitle={6th International Conference on AI in Computational Linguistics},
  year={2024}
}
Contact
For any questions or collaboration inquiries, please contact:

Mohammed Rasol Al Saidat: mohammedrasol@gmail.com
