import socket

def main():
# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
# Bind the socket to a specific address and port
server_socket.bind(('localhost', 12345))

# Listen for incoming connections
server_socket.listen(5)

print("Server is listening on port 12345...")

while True:
# Accept a connection from a client
client_socket, addr = server_socket.accept()

print(f"Connected by {addr}")

# Receive data from the client
data = client_socket.recv(1024).decode('utf-8')

if not data:
break

print(f"Received: {data}")

# Send a response back to the client
response = "Hello, client!"
client_socket.sendall(response.encode('utf-8'))

# Close the client socket
client_socket.close()

except KeyboardInterrupt:
print("Server is shutting down...")
finally:
# Close the server socket
server_socket.close()

if __name__ == "__main__":
main()
```