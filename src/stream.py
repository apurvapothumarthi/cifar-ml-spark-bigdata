#! /usr/bin/python3

import time
import json
import pickle
import socket
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(
    description='Streams a file to a Spark Streaming Context')
parser.add_argument('--file', '-f', help='File to stream', required=False,
                    type=str, default="cifar")    # path to file for streaming
parser.add_argument('--batch-size', '-b', help='Batch size',
                    required=False, type=int, default=100)  # default batch_size is 100
parser.add_argument('--endless', '-e', help='Enable endless stream',
                    required=False, type=bool, default=False)  # looping disabled by default
parser.add_argument('--train-count', '-c', help='Training batches count',
                    required=False, type=int, default=5)  # default is 5
parser.add_argument('--testing', '-t', help='Enable testing stream',
                    required=False, type=bool, default=False)  

TCP_IP = "localhost"
TCP_PORT = 6100


def connectTCP():   # connect to the TCP server -- there is no need to modify this function
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    print(f"Waiting for connection on port {TCP_PORT}...")
    connection, address = s.accept()
    print(f"Connected to {address}")
    return connection, address


# separate function to stream CIFAR batches since the format is different
def sendCIFARBatchFileToSpark(tcp_connection, input_batch_file):
    # load the entire dataset
    with open(input_batch_file, 'rb') as batch_file:
        batch_data = pickle.load(batch_file, encoding='bytes')

    # obtain image data and labels
    data = batch_data[b'data']
    data = list(map(np.ndarray.tolist, data))
    labels = batch_data[b'labels']
    # setting feature size to form the payload later
    feature_size = len(data[0])
    # iterate over batches of size batch_size
    for image_index in tqdm(range(0, len(data)-batch_size+2, batch_size)):
        # load batch of images
        image_data_batch = data[image_index:image_index+batch_size]
        image_label = labels[image_index:image_index +
                             batch_size]        # load batch of labels
        payload = dict()
        for mini_batch_index in range(len(image_data_batch)):
            payload[mini_batch_index] = dict()
            for feature_index in range(feature_size):  # iterate over features
                payload[mini_batch_index][f'feature{feature_index}'] = image_data_batch[mini_batch_index][feature_index]
            payload[mini_batch_index]['label'] = image_label[mini_batch_index]
        # print(payload)    # uncomment to see the payload being sent
        # encode the payload and add a newline character (do not forget the newline in your dataset)
        send_batch = (json.dumps(payload) + '\n').encode()
        try:
            tcp_connection.send(send_batch)  # send the payload to Spark
        except BrokenPipeError:
            print("Either batch size is too big for the dataset or the connection was closed")
        except Exception as error_message:
            print(f"Exception thrown but was handled: {error_message}")
        time.sleep(5)


def streamCIFARDataset(tcp_connection, input_folder='cifar',batch_count=5, test = False):
    print("Starting to stream CIFAR data")
    CIFAR_BATCHES = [
        'data_batch_1',
        'data_batch_2',   # uncomment to stream the second training dataset
        'data_batch_3',   # uncomment to stream the third training dataset
        'data_batch_4',   # uncomment to stream the fourth training dataset
        'data_batch_5',    # uncomment to stream the fifth training dataset
         'test_batch'      # uncomment to stream the test dataset
    ]
    if test:
        sendCIFARBatchFileToSpark(tcp_connection,input_folder+CIFAR_BATCHES[-1])
    else:
        for batch in CIFAR_BATCHES[0:batch_count]:
            sendCIFARBatchFileToSpark(tcp_connection,input_folder+batch)
            time.sleep(5)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    input_file = args.file
    batch_size = args.batch_size
    endless = args.endless
    train_count = args.train_count
    testing = args.testing
    tcp_connection, _ = connectTCP()

    if endless:
        while True:
            streamCIFARDataset(tcp_connection, input_file)
    else:
        streamCIFARDataset(tcp_connection, input_file,train_count,testing)

    tcp_connection.close()