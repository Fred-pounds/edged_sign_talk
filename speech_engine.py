import pyttsx3
import threading

class SpeechEngine:
    """
    Handles text-to-speech operations using pyttsx3.
    Designed to run speech in a way that doesn't block the main video loop efficiently.
    """
    def __init__(self):
        """Initialize the speech engine."""
        self.engine = pyttsx3.init()
        # Set properties if needed, e.g., rate or volume
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        
        # We need to run the engine loop. For pyttsx3, runAndWait is blocking.
        # To make it non-blocking for the main loop, we can usually just call say() 
        # and runAndWait() in a separate thread or use the engine's startLoop in a thread.
        # However, repeatedly creating threads for each word is safe enough for low frequency.
        # A more robust approach only for 'saying' one thing at a time:
        self.lock = threading.Lock()

    def say(self, text):
        """
        Speak the given text.
        
        Args:
            text (str): The text to convert to speech.
        """
        # Run in a separate thread to prevent blocking the video processing loop
        t = threading.Thread(target=self._speak_thread, args=(text,))
        t.daemon = True
        t.start()

    def _speak_thread(self, text):
        """Internal method to run the speech command in a thread."""
        with self.lock:
            # We initialize a new engine instance per thread if the global one has issues 
            # with threading, but pyttsx3 generally shares the driver. 
            # However, for simple usage:
            try:
                # Re-initializing inside thread is sometimes safer for certain drivers,
                # but let's try using the shared instance first.
                # Note: runAndWait() starts the event loop, iterates, and returns.
                self.engine.say(text)
                self.engine.runAndWait()
            except RuntimeError:
                # If loop is already running or other loop issues
                pass
            except Exception as e:
                print(f"Speech error: {e}")

    def cleanup(self):
        """Cleanup resources."""
        self.engine.stop()
