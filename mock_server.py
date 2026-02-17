import http.server
import socketserver
import time
import cv2
import numpy as np

# A simple MJPEG server for testing
class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            # Create a simple dummy image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            try:
                while True:
                    # Draw a moving square
                    frame = img.copy()
                    t = int(time.time() * 100) % 600
                    cv2.rectangle(frame, (t, 200), (t+50, 250), (0, 255, 0), -1)
                    
                    _, jpeg = cv2.imencode('.jpg', frame)
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                    time.sleep(0.05)
            except Exception as e:
                print(f"Client disconnected: {e}")

def run_server(port=8080):
    with socketserver.TCPServer(("", port), MJPEGHandler) as httpd:
        print(f"Mock MJPEG server running at http://localhost:{port}/stream")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
