import lcm
import time
import sys
import os

# Add current directory to path to import lcm_types
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lcm_types.task_msgs import task_command_t, task_result_t

class TaskTester:
    def __init__(self):
        self.lc = lcm.LCM()
        self.lc.subscribe("TASK_RESULT", self.handle_result)
        self.result_received = False
        self.last_success = 0

    def handle_result(self, channel, data):
        msg = task_result_t.decode(data)
        print(f"Received result: success={msg.success}")
        self.last_success = msg.success
        self.result_received = True

    def send_task(self, task_id):
        msg = task_command_t()
        msg.task_id = task_id
        self.lc.publish("TASK_COMMAND", msg.encode())
        print(f"Sent task command: {task_id}")
        self.result_received = False

    def wait_for_result(self, timeout=60):
        start_time = time.time()
        while not self.result_received:
            self.lc.handle_timeout(100)
            if time.time() - start_time > timeout:
                print("Timeout waiting for result")
                return False
        return True

def main():
    tester = TaskTester()
    
    while True:
        print("\nAvailable tasks:")
        print("1: pick_kettle")
        print("2: place_kettle")
        print("3: open_and_close")
        print("4: watering_flowers")
        print("q: Quit")
        
        choice = input("Enter task ID: ").strip().lower()
        
        if choice in ['q', 'quit', 'exit']:
            break
            
        if choice not in ['1', '2', '3', '4']:
            print("Invalid choice")
            continue
            
        task_id = int(choice)
        tester.send_task(task_id)
        
        print("Waiting for task completion...")
        if tester.wait_for_result(timeout=120): # 2 minutes timeout
            print(f"Task completed. Success: {tester.last_success}")
        else:
            print("Task timed out or no response received")

if __name__ == "__main__":
    main()
