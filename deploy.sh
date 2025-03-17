#!/bin/bash

# Kill any existing processes on ports 3000-3010 (frontend) and 8000 (backend)
echo "Cleaning up existing processes..."
lsof -ti:3000-3010 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Kill any existing uvicorn processes
pkill -f "uvicorn" 2>/dev/null || true

# Kill any existing react-scripts processes
pkill -f "react-scripts" 2>/dev/null || true

# Wait a moment for processes to clean up
sleep 2

# Start backend server
echo "Starting backend server..."
cd backend
source ../venv_py310/bin/activate
python -m uvicorn app.main:app --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend server
echo "Starting frontend server..."
cd ../frontend
npm start &
FRONTEND_PID=$!

# Function to handle cleanup on script termination
cleanup() {
    echo "Cleaning up processes..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait 