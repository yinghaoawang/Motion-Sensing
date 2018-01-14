# Webcam Security
Use your own computer's webcam to monitor security. It records whenever it detects motion.
To save space, it will automatically stop recording if there has been no motion for a short amount of time.

![Sample](https://i.imgur.com/HOCWlnB.png)

## Getting Started
Jar file download link [here](http://www.mediafire.com/file/n6fmr4zl2bpr3cc/cvt.zip)

### Using just the application
1. Download the jar zip
2. Unzip the zip file
3. Go into the directory and run cvt.jar

### Using IntelliJ IDEA
1. Clone the git repo
2. Open the main directory with IntelliJ IDEA
3. Build and run

## FAQ
- How do I begin recording?
    - It automatically enables recording when a file directory is selected. Recording begins when it detects motion. The word 'Rec' appears on the screen when it is recording.
- Where are the video files stored?
    - They are located in the directory as the JAR file

## Built With
* [JavaCV](https://github.com/bytedeco/javacv) - Java port to OpenCV (open source library for computer vision, image processing, and machine learning)
* [Gradle](https://gradle.org) - Build automation tool
* [IntelliJ IDEA](https://www.jetbrains.com/idea/) Java IDE
