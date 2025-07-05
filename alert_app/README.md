# Driver Drowsiness Detection App

A Flutter application that connects to a Flask backend for real-time driver drowsiness detection.

## Features

- Real-time drowsiness monitoring
- Audio alerts when drowsiness is detected
- Beautiful UI with status indicators
- Manual alarm control
- Connection status monitoring

## Setup Instructions

### 1. Install Dependencies

Make sure you have Flutter installed and run:

```bash
flutter pub get
```

### 2. Update Server IP

In `lib/main.dart`, update the `serverUrl` variable with your Flask server's IP address:

```dart
String serverUrl = "http://YOUR_SERVER_IP:5000";
```

### 3. Ensure Audio File

Make sure `alarm.mp3` is in the `assets/` folder.

### 4. Run the App

```bash
flutter run
```

## Backend Requirements

Your Flask backend should have:
- A `/status` endpoint that returns JSON with `{"drowsy": boolean}`
- Running on port 5000
- CORS enabled for mobile access

## Usage

1. Start your Flask backend server
2. Launch the Flutter app
3. Click "Start Monitoring" to begin drowsiness detection
4. The app will show alerts and play audio when drowsiness is detected
5. Use "Stop Alarm" to manually stop the audio alert

## Troubleshooting

- If you get connection errors, check that your Flask server is running and accessible
- Update the server IP address in the code to match your network
- Ensure the alarm.mp3 file is properly included in assets 