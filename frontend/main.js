// frontend/main.js

const { app, BrowserWindow, screen } = require('electron');
const path = require('path');

// --- Configuration ---
const INITIAL_WINDOW_WIDTH = 600; 
const INITIAL_WINDOW_HEIGHT = 700; 

function createWindow() {
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.workAreaSize;

    const mainWindow = new BrowserWindow({
        width: INITIAL_WINDOW_WIDTH,
        height: INITIAL_WINDOW_HEIGHT,
        x: Math.floor((width - INITIAL_WINDOW_WIDTH) / 2),
        y: Math.floor((height - INITIAL_WINDOW_HEIGHT) / 2),
        
        frame: true, // Enable standard window frame
        resizable: true, // Allows resizing
        transparent: false, 
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        }
    });

    mainWindow.loadFile(path.join(__dirname, 'index.html'));
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});