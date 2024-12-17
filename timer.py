import time

class Timer:
    def __init__(self, ndates, type="date"):
        self.ndates = ndates  # Number of steps/dates to process
        self.start_time = None  # Start time of the timer
        self.last_step = None   # Time of the last step
        self.type = type

    def start(self):
        """
        Initialize the timer and mark the starting point.
        """
        self.start_time = time.time()
        self.last_step = self.start_time

    def step(self, i=None):
        """
        Track progress for step `i` and display elapsed/estimated times.
        
        Args:
            i (int): Current step index (0-based).
        """
        if self.start_time is None:
            raise ValueError("Timer not started. Call start_timer() first.")
        
        now = time.time()  # Current time
        total_elapsed_time = now - self.start_time  # Total time elapsed
        step_elapsed_time = now - self.last_step  # Time since the last step

        # Update the last_step to the current time
        self.last_step = now
        
        # Calculate progress
        if i is not None:
            processed_percentage = i / self.ndates
            estimated_total_time = total_elapsed_time / processed_percentage if processed_percentage > 0 else 0
            remaining_time = estimated_total_time - total_elapsed_time
            estimated_finish = time.strftime('%Y-%m-%d %H:%M:%S', 
                                             time.localtime(self.start_time + estimated_total_time))
            
            # Print progress information
            print(f"Processing {self.type} {i+1}/{self.ndates}:\t"
                  f"Elapsed: {total_elapsed_time:.2f}s (+{step_elapsed_time:.2f}s),\t"
                  f"Remaining: {remaining_time:.2f}s,\t"
                  f"Estimated Finish: {estimated_finish}")
        else:
            print(f"Step completed. Time since last step: {step_elapsed_time:.2f}s")
    
    def total_time(self):
        if self.start_time is None:
            raise ValueError("Timer not started. Call start_timer() first.")
        
        now = time.time() 
        total_elapsed_time = now - self.start_time
        return total_elapsed_time
