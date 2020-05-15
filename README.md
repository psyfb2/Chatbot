# Chatbot #
Conversation AI Chatbot with a focus on keeping a consistent set of personality facts throughout conversation.
Models trained include:
* Sequence to Sequence
* Transformer
* Multiple Encoders  

This project consists of a Flask Website and Command Line Interface.

## Installation ##
To install the Command Line Interface, first install python3 and then use the following steps:  
```
git clone https://github.com/psyfb2/Chatbot.git
cd Chatbot/chatbot
pip install -r requirements.txt
```

## Command Line Interface ##
The CLI is responsible for training, evaluating and interacting with the models.  
Model choices include:  
* seq2seq
* deep_seq2seq
* multiple_encoders 
* deep_multiple_encoders
* transformer

One of these should be passed to either the train, eval or talk arguments.  

|      Argument Name      |                            Description                           |   Default Value   |
|:-----------------------:|:----------------------------------------------------------------:|:-----------------:|
| train                   | Name of the model to train                                       | None              |
| batch_size              | Training batch size                                              | 64                |
| epochs                  | Max number of training epochs                                    | 100               |
| early_stopping_patience | Number of epochs to run without best validation loss decreasing  | 7                 |
| segment_embedding       | Use segment embedding?                                           | True              |
| perform_pretraining     | pretrain models on Movie and Daily Dialog datasets?              | False             |
| verbose                 | Display loss for each batch?                                     | 0                 |
| min_epochs              | Number of epochs to run regardless of early stopping             | 30                |
| glove_filename          | Name of the glove file to use in the data folder                 | glove.6B.300d.txt |
|                         |                                                                  |                   |
| eval                    | Name of the model to evaluate  using Perplexity and F1           | None              |
|                         |                                                                  |                   |
| talk                    | Name of the model to interact with                               | None              |
| beam_width              | Beam width to use in beam search                                 | 3                 |
| beam_search             | Use beam search?                                                 | True              |
| plot_attention          | Plot attention weights, requires beam_search to be false.        | False             |
  
The CLI is at the directory chatbot/models/main.py 
   
Example usage to train Seq2Seq model:  
`python main.py --train seq2seq`  
Example usage to evaluate Multiple Encoders model:  
`python main.py --eval multiple_encoders`  
Example usage to interact with Transformer model:  
`python main.py --talk transformer`  
  
Note that trained models are not included in this github repo because of their large size.  
Trained models can be found here, just put them in the chatbot/saved_models folder:  
[Trained Models](https://drive.google.com/open?id=1WSH6bVltpNn78O7rBeFLWLExu-Zk5utP "Trained Models")  
  
Evaluation results can be found at chatbot/models/results.txt  
  
## Website ##
The models are deployed using a Flask backend.  
The live version can be found here:  
[Live Version](https://cloud.google.com/appengine/docs/standard/python3/building-app/deploying-web-service "Live Version")

