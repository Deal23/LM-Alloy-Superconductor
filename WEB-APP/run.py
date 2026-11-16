#!/usr/bin/env python3
"""
Entry point for the lightweight LMSC web app.
Usage: python run.py [--host 0.0.0.0] [--port 5050] [--debug]
"""

import argparse
import os
import socket
import subprocess
import sys
import threading
import webbrowser

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, PROJECT_ROOT)

from app import create_app


def _port_in_use(host, port):
    """Check whether a TCP port is already listening on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        target = "127.0.0.1" if host == "0.0.0.0" else host
        return s.connect_ex((target, port)) == 0


def main():
    parser = argparse.ArgumentParser(description="LMSC lightweight web app")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: local only)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "5050")),
        help="bind port (default: 5050; env: PORT)",
    )
    parser.add_argument("--debug", action="store_true", help="enable Flask debug mode")
    parser.add_argument("--no-browser", action="store_true", help="do not open browser")
    args = parser.parse_args()

    app = create_app()

    if _port_in_use(args.host, args.port):
        print(f"[错误] 端口 {args.port} 已被占用。")
        print("请使用其他端口启动，例如：python run.py --port 8080")
        print("或设置环境变量：PORT=8080 python run.py")
        sys.exit(1)

    if not args.no_browser:
        def open_browser():
            try:
                webbrowser.open(f"http://127.0.0.1:{args.port}")
            except Exception:
                pass
        threading.Timer(1.2, open_browser).start()

    print("=" * 50)
    print("LMSC 轻量版已启动")
    print(f"地址: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止")
    print("=" * 50)

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"[错误] 端口 {args.port} 启动时被占用：{e}")
            print("请使用：python run.py --port <其他端口>")
            sys.exit(1)
        raise


if __name__ == "__main__":
    if os.environ.get("LMSC_SUPERVISED") == "1":
        main()
    else:
        env = os.environ.copy()
        env["LMSC_SUPERVISED"] = "1"
        while True:
            try:
                result = subprocess.call([sys.executable, "-B", __file__, *sys.argv[1:]], env=env)
            except KeyboardInterrupt:
                break
            if result != 75:
                sys.exit(result)
            print("更新完成，正在重启服务……", flush=True)
