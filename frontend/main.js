// frontend/main.js
const { app, BrowserWindow, Menu, screen } = require('electron');
const path = require('path');

const INITIAL_WINDOW_WIDTH = 600;
const INITIAL_WINDOW_HEIGHT = 700;

function createWindow() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;

  const win = new BrowserWindow({
    width: INITIAL_WINDOW_WIDTH,
    height: INITIAL_WINDOW_HEIGHT,
    x: Math.floor((width - INITIAL_WINDOW_WIDTH) / 2),
    y: Math.floor((height - INITIAL_WINDOW_HEIGHT) / 2),

    // ✅ Keep native title bar & controls
    frame: true,
    resizable: true,
    backgroundColor: '#000',

    // (optional) On Windows, this hides menu bar if it ever existed
    // and prevents Alt from showing it
    autoHideMenuBar: true,

    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  // ✅ Remove File/Edit/View menu entirely
  Menu.setApplicationMenu(null);
  win.setMenuBarVisibility(false);

  win.loadFile(path.join(__dirname, 'index.html'));

  // win.webContents.openDevTools(); // <- uncomment for debugging
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  // Quit on all platforms except macOS (standard Electron behavior)
  if (process.platform !== 'darwin') app.quit();
});

// Clean up unhandled promise warnings in the console
process.on('unhandledRejection', (reason) => {
  console.warn('⚠️ Unhandled rejection:', reason);
});
