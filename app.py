"""
EcoBite AI - Streamlit Dashboard Entry Point
Run with:
    streamlit run app.py
or simply:
    python app.py
"""
import sys
import subprocess
from pathlib import Path

def is_running_under_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

dashboard_file = Path(__file__).resolve().parent / "dashboard" / "app.py"

if __name__ == "__main__":
    if is_running_under_streamlit():
        import runpy
        runpy.run_path(str(dashboard_file), run_name="__main__")
    else:
        print("\n========================================================")
        print("   Starting EcoBite AI Streamlit Dashboard (Dark Mode)")
        print("========================================================")
        print("Launching browser at http://localhost:8501 ...\n")
        
        # Check if .venv streamlit exists, else use current environment's streamlit
        venv_streamlit = Path(__file__).resolve().parent / ".venv" / "Scripts" / "streamlit.exe"
        if venv_streamlit.exists():
            streamlit_bin = str(venv_streamlit)
        else:
            streamlit_bin = sys.executable.replace("python.exe", "streamlit.exe")
            if not Path(streamlit_bin).exists():
                streamlit_bin = "streamlit"
                
        cmd = [streamlit_bin, "run", str(dashboard_file)]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
