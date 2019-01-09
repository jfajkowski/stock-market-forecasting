import logging
import socket

TCP_HOST = "127.0.0.1"
TCP_PORT = 10000
BUFFER_SIZE = 1024

logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(name)s: %(message)s', level=logging.INFO)

# Serve it on TCP socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.connect((TCP_HOST, TCP_PORT))

    logging.info('Connected to model running on %s:%s', TCP_HOST, TCP_PORT)

    model_input_line = '{"indexHistory":[{"date":"2016-06-30","openValue":2.0,"closeValue":3.0},{"date":"2016-07-01","openValue":1.0,"closeValue":2.0}],"articles":[{"date":"2016-06-29","header":"Test header 3"},{"date":"2016-06-30","header":"Test header 2"},{"date":"2016-07-01","header":"Test header 1"}]}'
    s.sendall(model_input_line.encode())
    logging.info('Sent %s', model_input_line)

    data = b''
    while True:
        buffer = s.recv(BUFFER_SIZE)
        data += buffer
        if len(buffer) < BUFFER_SIZE:
            break

    model_output_line = data.decode()
    logging.info('Received %s', model_output_line)
except Exception as e:
    logging.error(e)