"""
Run everything in one go: start the dashboard and process the streams in the
background. Results show up in the dashboard live as they are found.

Run:
    .venv/bin/python main.py [config.yaml]
"""

import threading

import uvicorn

import process
import dashboard  # reads the same config path from the command line


def main():
    dc = dashboard.CFG.get("dashboard", {})
    host, port = dc.get("host", "0.0.0.0"), int(dc.get("port", 9000))

    worker = threading.Thread(target=process.main, daemon=True)
    worker.start()

    print(f"Dashboard: http://localhost:{port}")
    uvicorn.run(dashboard.app, host=host, port=port)


if __name__ == "__main__":
    main()
