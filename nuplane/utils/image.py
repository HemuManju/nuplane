import socket
import numpy as np


def receive_image_over_tcp(image_width, image_height, channels, server_port):
    # Server configuration
    SERVER_IP = "127.0.0.1"  # Change this to your server IP
    SERVER_PORT = int(server_port)  # Change this to your server port
    image_width = int(image_width)
    image_height = int(image_height)

    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind socket to address
    try:
        server_socket.bind((SERVER_IP, SERVER_PORT))

        # Listen for incoming connections
        server_socket.listen(1)

        # Accept incoming connection
        client_socket, client_address = server_socket.accept()

        # Receive image data
        received_data = b""
        while len(received_data) < image_width * image_height * channels:
            chunk = client_socket.recv(65000)
            if not chunk:
                break
            received_data += chunk

        # Convert received data to NumPy array
        image_array = np.frombuffer(received_data, dtype=np.uint8)
        image_array = image_array.reshape((image_height, image_width, channels))

        # Close sockets
        client_socket.close()
        server_socket.close()
    except OSError:
        image_array = None

    return image_array
