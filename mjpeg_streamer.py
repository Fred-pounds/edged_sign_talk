import cv2
import requests
import numpy as np
import threading
import time

class MJPEGStreamer:
    """
    Robust MJPEG stream client that runs in a separate thread.
    Ensures the main recognition loop always gets the freshest frame.
    """
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.lock = threading.Lock()
        
    def start(self):
        print(f"Starting MJPEG stream from: {self.url}")
        self.thread.start()
        return self

    def _update(self):
        try:
            stream = requests.get(self.url, stream=True, timeout=10)
            if stream.status_code != 200:
                print(f"Failed to connect to stream: {stream.status_code}")
                return

            bytes_buffer = bytes()
            for chunk in stream.iter_content(chunk_size=1024):
                if self.stopped:
                    break
                
                bytes_buffer += chunk
                a = bytes_buffer.find(b'\xff\xd8')
                b = bytes_buffer.find(b'\xff\xd9')
                
                if a != -1 and b != -1:
                    jpg = bytes_buffer[a:b+2]
                    bytes_buffer = bytes_buffer[b+2:]
                    
                    # Decode image
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        with self.lock:
                            self.frame = img
                            
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            self.stopped = True

    def read(self):
        """Returns the latest frame and a success boolean."""
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)

    def release(self):
        self.stop()

    def is_opened(self):
        return not self.stopped

    def isOpened(self):
        return self.is_opened()
