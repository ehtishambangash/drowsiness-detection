# Drowsiness Detection System

## Overview
The **Drowsiness Detection System** is an AI-powered application designed to detect driver drowsiness in real-time using computer vision. By analyzing facial landmarks, such as eye closure and head position, the system provides timely alerts to enhance road safety.

## Features
- **Real-time Detection** – Uses advanced facial recognition to identify drowsiness.  
- **Instant Alerts** – Provides both visual and audible warnings to prevent accidents.  
- **Lightweight & Efficient** – Optimized for smooth real-time performance.  
- **Cross-Platform Support** – Works on **Windows, Linux, and macOS**.  
- **Customizable Sensitivity** – Adjustable settings for different environments.  

## How It Works
1. **Face Detection** – The system detects the driver’s face using OpenCV and dlib.  
2. **Eye & Head Analysis** – It tracks eye closure duration and head position.  
3. **Drowsiness Detection** – If the eyes remain closed beyond a threshold, an alert is triggered.  
4. **Alarm System** – An audible and visual warning is given to wake up the driver.  

## Model File
This project requires the **shape_predictor_68_face_landmarks.dat** file for facial landmark detection. You can download it from the following link:
[Download shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

After downloading, place it in the project directory before running the application.

## Contact
**Author**: Ehtisham Bangash  
For inquiries, contact me at ehtishambangash111@gmail.com or [GitHub](https://github.com/ehtishambangash)

