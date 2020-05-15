# -*- coding: utf-8 -*-
"""
@author: Fady Benattayallah
"""
from flask import Flask, render_template, jsonify, request
import logging
import sys
import os
import tensorflow as tf
path = os.path.abspath(os.path.join(os.path.dirname( __file__ )))
path = os.path.join(path, 'chatbot', 'models')
sys.path.insert(1, path)
import seq2seq_model
import transformer
import multiple_encoders
from text_preprocessing import clean_line

app = Flask(__name__)

def load_models():
    global transformer_chatbot
    global multiple_encoders_chatbot
    global seq2seq_chatbot
    
    # tf.functions cause retracing, which means flask server will restart
    # disable this behaviour
    # in a perfect world these models should be behind a Tensorflow Serving
    # but alas we lack Google Cloud Funds to do this :(
    tf.config.experimental_run_functions_eagerly(True)
    
    transformer_chatbot       = transformer.ChatBot()
    multiple_encoders_chatbot = multiple_encoders.ChatBot(deep_model=False)
    seq2seq_chatbot           = seq2seq_model.ChatBot(deep_model=False)
    
load_models()

@app.route('/_get_reply')
def get_reply():
    model_name = request.args.get('model_name', 'seq2seq', type=str)
    message = request.args.get('message', '__silence__', type=str)
    persona = request.args.get('persona', 
              'i like to exercise . i like to listen to music . i have a boxer dog . i like baths .',
              type=str)
    
    message = clean_line(message)
    
    if model_name == 'multiple_encoders':
        reply = multiple_encoders_chatbot.reply(persona, message)
    elif model_name == 'transformer':
        reply = transformer_chatbot.reply(persona, message)
    else:
        reply = seq2seq_chatbot.reply(persona, message)
    
    return jsonify(result=reply)


@app.route('/')
def index():
    return render_template('index.html')


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # this is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
