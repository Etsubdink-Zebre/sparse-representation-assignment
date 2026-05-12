#!/usr/bin/env python3
"""
Simple script to open the dashboard in the default browser
"""
import webbrowser
import os
import sys

def open_dashboard():
    """Open the dashboard in the default browser"""
    # Get the absolute path to the dashboard
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html')
    
    # Convert to file:// URL
    file_url = f'file:///{dashboard_path.replace(os.sep, "/")}'
    
    print(f"Opening dashboard: {file_url}")
    
    try:
        webbrowser.open(file_url)
        print("Dashboard opened in your default browser!")
        print("If the dashboard doesn't load properly, make sure you're opening dashboard.html directly")
    except Exception as e:
        print(f"Error opening browser: {e}")
        print(f"Please open this file manually: {dashboard_path}")

if __name__ == "__main__":
    open_dashboard()
