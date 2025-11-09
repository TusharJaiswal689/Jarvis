// frontend/main.js

const { app, BrowserWindow, screen } = require('electron');
const path = require('path');

// --- Configuration ---
// Adjust dimensions for the minimalist circular UI
const BASE_SIZE = 400; 

function createWindow() {
    // 1. Get the primary display bounds to calculate the center position
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.workAreaSize;

    const mainWindow = new BrowserWindow({
        width: BASE_SIZE,
        height: BASE_SIZE,
        x: Math.floor((width - BASE_SIZE) / 2), // Center X
        y: Math.floor((height - BASE_SIZE) / 2), // Center Y
        
        frame: false, // CRUCIAL for frameless, minimalist UI
        resizable: false,
        transparent: true, // CRUCIAL for transparent background
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        }
    });

    mainWindow.loadFile(path.join(__dirname, 'index.html'));

    // Note: We only use DevTools in development.
    // mainWindow.webContents.openDevTools(); 
}

// When Electron is ready to create browser windows.
app.whenReady().then(() => {
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

// Quit when all windows are closed.
app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});