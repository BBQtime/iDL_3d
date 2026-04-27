#!/bin/bash

# 1. Path Definitions
PYTHON_EXEC="/home/alan/anaconda3/envs/py38/bin/python"
APP_PATH="/mnt/faststorage/alan/iDL_3d/py_code/main_ui.py"
DISPLAY_NUM=":1"
LOCK_FILE="/tmp/.X${DISPLAY_NUM:1}-lock"

# 2. Elegant Cleanup
# Only kills the server if the lock file exists
if [ -f "$LOCK_FILE" ]; then
    echo "Existing session found on $DISPLAY_NUM. Terminating..."
    kasmvncserver -kill $DISPLAY_NUM > /dev/null 2>&1
    # Force remove stale sockets if they persist
    rm -f "/tmp/.X11-unix/X${DISPLAY_NUM:1}" "$LOCK_FILE"
else
    echo "No active session found on $DISPLAY_NUM. Proceeding..."
fi

# 3. Start KasmVNC
# Removed -geometry to allow the YAML and Browser to handle sizing
echo "Starting KasmVNC service on $DISPLAY_NUM..."
kasmvncserver $DISPLAY_NUM -geometry 1920x850

# 4. Wait for Initialization
sleep 2

# 5. Environment Setup and Execution
export DISPLAY=$DISPLAY_NUM
export PYTHONPATH="${PYTHONPATH}:/mnt/faststorage/alan/iDL_3d"

# Enable Qt Debugging to track library issues
export QT_DEBUG_PLUGINS=1

echo "Launching Qt UI application..."
$PYTHON_EXEC $APP_PATH