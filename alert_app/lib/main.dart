import 'package:flutter/material.dart';
// import 'package:audioplayers/audioplayers.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Driver Drowsiness Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: DrowsinessDetectionScreen(),
    );
  }
}

class DrowsinessDetectionScreen extends StatefulWidget {
  @override
  _DrowsinessDetectionScreenState createState() =>
      _DrowsinessDetectionScreenState();
}

class _DrowsinessDetectionScreenState extends State<DrowsinessDetectionScreen> {
  // final AudioPlayer audioPlayer = AudioPlayer();
  bool isAlarmPlaying = false;
  bool isMonitoring = false;
  bool isDrowsy = false;
  String statusMessage = "Not Monitoring";
  Timer? _timer;
  String serverUrl = "http://192.168.70.240:5000"; // Your computer's IP address

  @override
  void initState() {
    super.initState();
    // _initializeAudio();
  }

  // void _initializeAudio() async {
  //   try {
  //     await audioPlayer.setSource(AssetSource('alarm.mp3'));
  //   } catch (e) {
  //     print('Error loading audio: $e');
  //   }
  // }

  void _startMonitoring() {
    setState(() {
      isMonitoring = true;
      statusMessage = "Monitoring...";
    });

    // Poll the server every 500ms
    _timer = Timer.periodic(Duration(milliseconds: 500), (timer) {
      _checkDrowsiness();
    });
  }

  void _stopMonitoring() {
    setState(() {
      isMonitoring = false;
      statusMessage = "Not Monitoring";
      isDrowsy = false;
    });

    _timer?.cancel();
    _stopAlarm();
  }

  Future<void> _checkDrowsiness() async {
    try {
      final response = await http.get(
        Uri.parse('$serverUrl/status'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(Duration(seconds: 5));

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final bool drowsy = data['drowsy'] ?? false;

        setState(() {
          isDrowsy = drowsy;
          if (drowsy) {
            statusMessage = "DROWSINESS DETECTED!";
            if (!isAlarmPlaying) {
              _playAlarm();
            }
          } else {
            statusMessage = "Monitoring...";
            if (isAlarmPlaying) {
              _stopAlarm();
            }
          }
        });
      } else {
        setState(() {
          statusMessage = "Server Error";
        });
      }
    } catch (e) {
      setState(() {
        statusMessage = "Connection Error";
      });
      print('Error checking drowsiness: $e');
    }
  }

  void _playAlarm() async {
    try {
      // await audioPlayer.play(AssetSource('alarm.mp3'));
      setState(() {
        isAlarmPlaying = true;
      });
      print('ALARM: Drowsiness detected!');
    } catch (e) {
      print('Error playing alarm: $e');
    }
  }

  void _stopAlarm() async {
    try {
      // await audioPlayer.stop();
      setState(() {
        isAlarmPlaying = false;
      });
    } catch (e) {
      print('Error stopping alarm: $e');
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    // audioPlayer.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Driver Drowsiness Detection'),
        backgroundColor: isDrowsy ? Colors.red : Colors.blue,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: isDrowsy
                ? [Colors.red.shade100, Colors.red.shade200]
                : [Colors.blue.shade50, Colors.blue.shade100],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Status Icon
              Container(
                width: 200,
                height: 200,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isDrowsy ? Colors.red : Colors.green,
                  boxShadow: [
                    BoxShadow(
                      color: isDrowsy
                          ? Colors.red.withOpacity(0.3)
                          : Colors.green.withOpacity(0.3),
                      blurRadius: 20,
                      spreadRadius: 5,
                    ),
                  ],
                ),
                child: Icon(
                  isDrowsy ? Icons.warning : Icons.check_circle,
                  size: 100,
                  color: Colors.white,
                ),
              ),

              SizedBox(height: 40),

              // Status Text
              Text(
                statusMessage,
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: isDrowsy ? Colors.red.shade800 : Colors.green.shade800,
                ),
                textAlign: TextAlign.center,
              ),

              SizedBox(height: 20),

              // Monitoring Status
              Container(
                padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                decoration: BoxDecoration(
                  color: isMonitoring ? Colors.green : Colors.grey,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  isMonitoring ? "ACTIVE" : "INACTIVE",
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),

              SizedBox(height: 40),

              // Control Buttons
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton(
                    onPressed: isMonitoring ? null : _startMonitoring,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green,
                      padding:
                          EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    child: Text(
                      'Start Monitoring',
                      style: TextStyle(fontSize: 16, color: Colors.white),
                    ),
                  ),
                  ElevatedButton(
                    onPressed: isMonitoring ? _stopMonitoring : null,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                      padding:
                          EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    child: Text(
                      'Stop Monitoring',
                      style: TextStyle(fontSize: 16, color: Colors.white),
                    ),
                  ),
                ],
              ),

              SizedBox(height: 20),

              // Manual Alarm Stop Button
              if (isAlarmPlaying)
                ElevatedButton(
                  onPressed: _stopAlarm,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    padding: EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                  child: Text(
                    'Stop Alert',
                    style: TextStyle(fontSize: 16, color: Colors.white),
                  ),
                ),

              SizedBox(height: 40),

              // Instructions
              Container(
                padding: EdgeInsets.all(20),
                margin: EdgeInsets.symmetric(horizontal: 20),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.8),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Column(
                  children: [
                    Text(
                      'Instructions:',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 10),
                    Text(
                      '1. Make sure your Flask server is running\n'
                      '2. Click "Start Monitoring" to begin\n'
                      '3. The app will show visual alerts when drowsiness is detected\n'
                      '4. Audio alerts will be added later',
                      style: TextStyle(fontSize: 14),
                      textAlign: TextAlign.left,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
