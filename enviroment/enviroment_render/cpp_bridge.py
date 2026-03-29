import json
import socket
import subprocess
import atexit
import time
import os
import sys
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

    def _find_executable(self):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'cpp_render'))
        
        # Platform-specific paths
        if sys.platform == 'win32':
            paths = [
                os.path.join(base_path, 'build', 'Release', 'strategy_renderer.exe'),
                os.path.join(base_path, 'build', 'strategy_renderer.exe'),
                os.path.join(base_path, 'build', 'Release', 'strategy_renderer'),
                os.path.join(base_path, 'bin', 'strategy_renderer.exe'),
            ]
            ext = '.exe'
        else:
            paths = [
                os.path.join(base_path, 'build', 'Release', 'strategy_renderer'),
                os.path.join(base_path, 'build', 'strategy_renderer'),
                os.path.join(base_path, 'bin', 'strategy_renderer'),
            ]
            ext = ''
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        return None

    def _start_renderer(self):
        executable_path = self._find_executable()
        
        if not executable_path:
            print(f"WARNING: C++ renderer executable not found")
            print(f"Searched in: {os.path.join(os.path.dirname(__file__), '..', '..', 'cpp_render', 'build', '...')}")
            print("Please build the C++ renderer first.")
            return
        
        print(f"Starting C++ renderer: {executable_path}")
            
        # Start the C++ renderer as a separate process
        self.proc = subprocess.Popen([executable_path])
        
        # Establish TCP socket connection
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        time.sleep(2) # Give the C++ process a moment to start its server
        try:
            self.socket.connect(("127.0.0.1", 8080))
            print("Connected to C++ renderer on port 8080")
        except ConnectionRefusedError:
            print("ERROR: Could not connect to the C++ renderer. Is it running and listening?")
            self.proc.kill()
            self.proc = None


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
