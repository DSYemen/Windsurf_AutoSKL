import os
import sys
import time
import signal
import psutil
import subprocess

def kill_streamlit_processes():
    """Kill any running Streamlit processes"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'streamlit' in proc.info['name'].lower():
                os.kill(proc.info['pid'], signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def main():
    # Kill existing Streamlit processes
    kill_streamlit_processes()
    time.sleep(2)  # Wait for processes to terminate
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Set environment variable for development
    os.environ['PYTHONPATH'] = project_root
    
    # Start Streamlit app
    app_path = os.path.join(project_root, "app", "ui", "dashboard_new.py")
    subprocess.run(["streamlit", "run", app_path])

if __name__ == "__main__":
    main()
