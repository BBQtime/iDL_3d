#!/bin/bash

# 1. Path Definitions
PYTHON_EXEC="/home/alan/anaconda3/envs/py38/bin/python"
APP_PATH="/mnt/faststorage/alan/iDL_3d/py_code/main_ui.py"
DISPLAY_NUM=":1"
LOCK_FILE="/tmp/.X${DISPLAY_NUM:1}-lock"

# 2. Elegant Cleanup
if [ -f "$LOCK_FILE" ]; then
    echo "Existing session found. Cleaning up..."
    kasmvncserver -kill $DISPLAY_NUM > /dev/null 2>&1
    rm -f "/tmp/.X11-unix/X${DISPLAY_NUM:1}" "$LOCK_FILE"
fi

# 3. Start KasmVNC at 1080p
echo "Starting KasmVNC at 1920x1080..."
kasmvncserver $DISPLAY_NUM -geometry 1920x1080

# 4. Wait for X11 to stabilize
sleep 3

# 5. The "Hammer": Force switch to the existing 1080p mode
export DISPLAY=$DISPLAY_NUM
echo "Forcing resolution to 1920x1080..."

xrandr -s 1920x1080 || echo "Switch failed, check your xrandr list."

# 6. Environment Setup and Execution
export PYTHONPATH="${PYTHONPATH}:/mnt/faststorage/alan/iDL_3d"
export QT_DEBUG_PLUGINS=1

echo "Launching Qt UI..."
$PYTHON_EXEC $APP_PATH