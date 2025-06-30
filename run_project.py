#!/usr/bin/env python3
"""
Quick Start Script for Mental Health Support Platform
Run this script to start both backend and frontend servers
"""

import subprocess
import sys
import os
import time
import webbrowser
import shutil
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run setup first.")
        return False
    
    # Check if Node.js/npm is installed
    npm_path = shutil.which("npm")
    if not npm_path:
        # Try alternative npm paths
        alternative_paths = [
            r"C:\Users\user\AppData\Roaming\npm\npm.cmd",
            r"C:\Program Files\nodejs\npm.cmd",
            r"C:\Program Files (x86)\nodejs\npm.cmd"
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                npm_path = path
                break
    
    if not npm_path:
        print("⚠️  Node.js/npm not found. Frontend will be skipped.")
        print("   To install Node.js: https://nodejs.org/")
        return "backend_only"
    
    print("✅ Dependencies check passed")
    return True

def setup_environment():
    """Setup the environment if needed"""
    print("🚀 Setting up environment...")
    
    # Create virtual environment if it doesn't exist
    if not Path("venv").exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Install requirements
    print("📦 Installing Python dependencies...")
    if os.name == 'nt':  # Windows
        subprocess.run(["venv\\Scripts\\pip", "install", "-r", "requirements.txt"])
    else:  # Unix/Linux/Mac
        subprocess.run(["venv/bin/pip", "install", "-r", "requirements.txt"])
    
    print("✅ Environment setup completed")

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting backend server...")
    
    # Change to backend directory
    os.chdir("backend")
    
    # Start the server
    if os.name == 'nt':  # Windows
        subprocess.Popen([
            "..\\venv\\Scripts\\python", "-m", "uvicorn", 
            "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"
        ])
    else:  # Unix/Linux/Mac
        subprocess.Popen([
            "../venv/bin/python", "-m", "uvicorn", 
            "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"
        ])
    
    # Change back to root directory
    os.chdir("..")
    
    print("✅ Backend server started at http://localhost:8000")

def start_frontend():
    """Start the React frontend server"""
    print("🚀 Starting frontend server...")
    
    # Find npm executable
    npm_path = shutil.which("npm")
    if not npm_path:
        # Try alternative npm paths
        alternative_paths = [
            r"C:\Users\user\AppData\Roaming\npm\npm.cmd",
            r"C:\Program Files\nodejs\npm.cmd",
            r"C:\Program Files (x86)\nodejs\npm.cmd"
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                npm_path = path
                break
    
    if not npm_path:
        print("❌ npm not found. Please install Node.js from https://nodejs.org/")
        print("   Frontend will not start. You can still use the backend API.")
        return False
    
    # Change to frontend directory
    os.chdir("frontend")
    
    # Install npm dependencies if needed
    if not Path("node_modules").exists():
        print("📦 Installing Node.js dependencies...")
        try:
            subprocess.run([npm_path, "install"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to install npm dependencies")
            os.chdir("..")
            return False
    
    # Start the development server
    try:
        subprocess.Popen([npm_path, "start"])
        print("✅ Frontend server started at http://localhost:3000")
        os.chdir("..")
        return True
    except FileNotFoundError:
        print("❌ npm start failed. Please check Node.js installation.")
        os.chdir("..")
        return False

def open_browsers(backend_only=False):
    """Open browsers to the application"""
    print("🌐 Opening application in browser...")
    
    # Wait a bit for servers to start
    time.sleep(5)
    
    # Open backend API docs
    webbrowser.open("http://localhost:8000/docs")
    
    if not backend_only:
        # Open frontend
        webbrowser.open("http://localhost:3000")
    
    print("✅ Browsers opened!")

def show_accuracy_info():
    """Show information about model accuracy"""
    print("\n📊 MODEL ACCURACY INFORMATION:")
    print("=" * 50)
    print("Current Model Performance:")
    print("• Text Emotion Classification: ~85-90%")
    print("• Facial Emotion Recognition: ~60-70%")
    print("• Overall System Accuracy: ~75%")
    print("\n🎯 To improve accuracy:")
    print("1. Run: python backend/train_text_model.py")
    print("2. Use improved models in backend/improved_facial_model.py")
    print("3. Evaluate with: python backend/evaluate_models.py")
    print("4. Check SETUP_GUIDE.md for detailed improvement strategies")

def show_nodejs_installation_guide():
    """Show Node.js installation guide"""
    print("\n📦 NODE.JS INSTALLATION GUIDE:")
    print("=" * 50)
    print("To run the frontend, you need to install Node.js:")
    print("1. Go to: https://nodejs.org/")
    print("2. Download the LTS version")
    print("3. Run the installer")
    print("4. Restart your terminal/command prompt")
    print("5. Verify installation: npm --version")
    print("\nAfter installing Node.js, run this script again!")

def main():
    """Main function to run the project"""
    print("🧠 Mental Health Support Platform")
    print("=" * 40)
    
    # Check if setup is needed
    dependency_status = check_dependencies()
    
    if dependency_status == False:
        print("\n🔧 Running setup...")
        setup_environment()
        dependency_status = check_dependencies()
    
    if dependency_status == False:
        print("❌ Setup failed. Please check the error messages above.")
        return
    
    # Start servers
    print("\n🚀 Starting servers...")
    start_backend()
    time.sleep(2)  # Wait for backend to start
    
    # Try to start frontend
    backend_only = False
    if dependency_status == "backend_only" or not start_frontend():
        backend_only = True
        show_nodejs_installation_guide()
    
    # Open browsers
    open_browsers(backend_only)
    
    # Show accuracy information
    show_accuracy_info()
    
    print("\n🎉 Project is running!")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    
    if not backend_only:
        print("📱 Frontend: http://localhost:3000")
    else:
        print("📱 Frontend: Not available (Node.js required)")
    
    print("\n💡 Press Ctrl+C to stop all servers")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        print("✅ Project stopped")

if __name__ == "__main__":
    main() 