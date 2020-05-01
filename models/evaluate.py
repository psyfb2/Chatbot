import tensorflow as tf
import text_preprocessing as pre
from abc import ABCMeta, abstractmethod

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')

def CCE_loss(label, pred):
    '''
    Calculate CCE loss for a batch of predictions without dividing by batch size
    '''
    # do not calculate loss for padding values
    mask = tf.math.logical_not(tf.math.equal(label, 0))
    loss_ = loss_object(label, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)

def f1(get_reply):
    ''' Get mean F1 score over test set '''
    persona, data = pre.load_dataset(pre.TEST_FN)
    
    f1 = 0
    for i in range(len(data)):
        pers  = persona[int(data[i, 2])]
        msg   = data[i, 0]
        reply = data[i, 1]
        
        pred = get_reply(pers, msg)
        # every word in pred should be seperated by space
        # including punctuation
        f1 += sentence_f1(reply, pred)
    return f1 / len(data)

def sentence_f1(reply, pred):
    # F1 = 2 * (precision * recall) / (precision + recall)
    # precision = # matched words / # pred words
    # recall = # matches words / # reply words
    
    reply = reply.split(' ')
    pred = pred.split(' ')
    
    matched_words = len([w for w in pred if w in reply])
    
    if matched_words == 0:
        return 0
    
    precision = matched_words / len(pred)
    recall = matched_words / len(reply)
    
    return 2 * ( (precision * recall) / (precision + recall) )


class BaseBot(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def reply(self, msg, persona):
        pass
    
    @abstractmethod
    def plot_attn(self):
        pass
    
    @abstractmethod
    def eval_f1(self):
        pass
    
    @abstractmethod
    def eval_ppl(self):
        pass