import json
import logging
import pickle
import socket

TCP_HOST = "127.0.0.1"
TCP_PORT = 10000
BUFFER_SIZE = 1024
QUEUE_SIZE = 128

logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(name)s: %(message)s', level=logging.INFO)

# Load serialized model
with open('./models/raw_bigram_svm.mdl', mode='rb') as vcr_file:
    model = pickle.load(vcr_file)

# Serve it on TCP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_HOST, TCP_PORT))
s.listen(QUEUE_SIZE)

logging.info('Model running on %s:%s', TCP_HOST, TCP_PORT)

while True:
    try:
        connection, address = s.accept()

        logging.info('%s:%s connected', *address)

        data = b''
        while True:
            buffer = connection.recv(BUFFER_SIZE)
            data += buffer
            if len(buffer) < BUFFER_SIZE:
                break

        model_input_line = data.decode()
        logging.info('%s:%s received %s', *address, model_input_line)
        model_input = json.loads(model_input_line)

        articles = sorted(model_input['articles'], key=lambda x: x['date'])

        raw = list(map(lambda x: x['header'], articles))
        y = model.predict([raw[-1]])
        model_output = y[0]
        model_output_line = str(list(model_output))
        connection.sendall(model_output_line.encode())
        logging.info('%s:%s sent %s', *address, model_output_line)
        connection.close()
    except Exception as e:
        logging.error(e)