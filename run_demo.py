import http.server
import socketserver
import webbrowser
import os

# Configuration
PORT = 8000
DIRECTORY = "docs"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"Serving at {url}")
        print("Press Ctrl+C to stop the server.")
        
        # Open browser automatically
        webbrowser.open(url)
        
        httpd.serve_forever()
except OSError as e:
    print(f"Error: {e}")
    print(f"Try running directly: python -m http.server --directory docs")
except KeyboardInterrupt:
    print("\nServer stopped.")
