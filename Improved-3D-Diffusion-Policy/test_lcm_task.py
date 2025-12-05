import lcm
import time
import struct

class TaskTester:
    def __init__(self):
        self.lc = lcm.LCM("udpm://239.255.50.50:10010?ttl=1")
        self.lc.subscribe("TASK_RESULT", self.handle_result)
        self.result_received = False
        self.last_success = 0

    def handle_result(self, channel, data):
        # 直接解码为int (小端序, 4字节)
        result = struct.unpack('<i', data)[0]
        print(f"Received result: success={result}")
        self.last_success = result
        self.result_received = True

    def send_task(self, task_id):
        # 直接编码int为字节流
        data = struct.pack('<i', task_id)
        self.lc.publish("TASK_COMMAND", data)
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
        print("2: open_and_close")
        print("3: watering_flowers")
        print("4: place_kettle")
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
