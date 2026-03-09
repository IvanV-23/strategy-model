import json
import socket
import subprocess
import atexit
import time
import os
import numpy as np

class CppRendererBridge:
    def __init__(self, width, height, metadata):
        self.width = width
        self.height = height
        self.metadata = metadata
        self.proc = None
        self.conn = None
        self.socket = None
        self._start_renderer()
        atexit.register(self.close)

    def _start_renderer(self):
        # Assumes the executable is built and located in cpp_render/build/Release
        executable_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cpp_render', 'build', 'Release', 'strategy_renderer.exe')
        
        if not os.path.exists(executable_path):
            print(f"WARNING: C++ renderer executable not found at {executable_path}")
            print("Please build it first (e.g., using CMake and Make/Ninja).")
            return
            
        # Start the C++ renderer as a separate process
        self.proc = subprocess.Popen([executable_path])
        
        # Establish TCP socket connection
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        time.sleep(1) # Give the C++ process a moment to start its server
        try:
            self.socket.connect(("127.0.0.1", 8080))
        except ConnectionRefusedError:
            print("ERROR: Could not connect to the C++ renderer. Is it running and listening?")
            self.proc.kill()


    def render_frame(self, render_mode, state_data):
        if self.socket is None:
            return
        
        # Convert all numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        try:
            # Serialize state_data to JSON and send
            json_payload = json.dumps(state_data, default=convert_numpy) + '\n'
            self.socket.sendall(json_payload.encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError):
            print("C++ renderer connection lost. Closing.")
            self.close()

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.proc:
            self.proc.kill()
            self.proc = None
