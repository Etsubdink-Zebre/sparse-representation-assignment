#!/usr/bin/env python3
"""
Simple HTTP server to serve the dashboard
"""
import http.server
import socketserver
import webbrowser
import os
import threading
import time

PORT = 8083

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def start_server():
    """Start the HTTP server"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 Server running at http://localhost:{PORT}")
        print(f"📊 Dashboard available at: http://localhost:{PORT}/dashboard.html")
        print("🔄 Press Ctrl+C to stop the server")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://localhost:{PORT}/dashboard.html')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
            httpd.shutdown()

if __name__ == "__main__":
    start_server()
