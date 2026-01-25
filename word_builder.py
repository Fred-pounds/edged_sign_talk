import time

class WordBuilder:
    """
    Manages the logic for stabilizing recognized letters and forming words.
    """
    def __init__(self, stability_duration=1.0):
        """
        Initialize the word builder.
        
        Args:
            stability_duration (float): Time in seconds a letter must be held to be accepted.
        """
        self.stability_duration = stability_duration
        self.current_word = ""
        self.last_letter = None
        self.stable_start_time = 0
        self.vocabulary = {"HELLO", "HI", "YES", "NO", "HELP"}

    def process_letter(self, letter):
        """
        Process a raw predicted letter. Returns the confirmed letter if stable, else None.
        
        Args:
            letter (str): The raw prediction from the model.
            
        Returns:
            str: The confirmed letter if added to current word, else None.
        """
        current_time = time.time()
        
        if letter == self.last_letter:
            # Letter is consistent
            if current_time - self.stable_start_time >= self.stability_duration:
                # Timer elapsed, confirm letter
                # Avoid adding the same letter repeatedly immediately after confirming it
                # Logic: We add it, then reset timer or valid state. 
                # To prevent endless 'AAAAAA', we can require a 'release' or just simple delay.
                # Project requirement: "accumulate letters into a word buffer". 
                # Assuming standard fingerspelling flow.
                
                # Check if we just added this letter to the end (optional debounce)
                # For simplicity, we add it. 
                # Implementing a "wait for change" might be better UI, but we stick to reqs:
                # "letter must remain stable for ~1 second before being accepted"
                
                # To prevent rapid fire of same letter once stable, we verify we haven't just consumed it.
                # A simple way is to force a reset of state after acceptance.
                self.current_word += letter
                self.last_letter = None # Reset so user has to sign again or hold again
                self.stable_start_time = 0 
                return letter
        else:
            # Letter changed, reset timer
            self.last_letter = letter
            self.stable_start_time = current_time
            
        return None

    def check_word(self):
        """
        Check if the current buffer matches any vocabulary word.
        
        Returns:
            str: The matched word if found, else None.
        """
        if self.current_word in self.vocabulary:
            matched_word = self.current_word
            self.clear() # Clear buffer on match
            return matched_word
        return None

    def get_current_word(self):
        """Get the current word buffer."""
        return self.current_word

    def clear(self):
        """Clear the word buffer."""
        self.current_word = ""
