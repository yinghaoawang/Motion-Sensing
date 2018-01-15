/*
 * Built upon Angelo Marchesin sample motion detector on javaCV sample files.
 * Yinghao Wang
 */

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.avcodec;
import org.bytedeco.javacv.*;

import javax.swing.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class MotionRecorder {
    private static final DateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd__HH-mm-ss");
    // Records webcam
    static FrameRecorder recorder = null;
    // Gets information and frames from webcam
    static OpenCVFrameGrabber grabber = null;

    // used to keep track of when to stop recording
    static int framesWithoutMotion = 0;
    static final int MAXFRAMESWITHOUTMOTION = 120;
    // used cross thread to determine if currently recording
    static volatile boolean recording = false;

    // the current image being recorded
    static IplImage live;
    // the frame that displays the recording
    static CanvasFrame liveFrame;

    // Does many of the conversions of the camera captures
    static OpenCVFrameConverter.ToIplImage converter;

    // Thread that runs all the webcam grabbing, recording, and motion sensing
    static Thread mainThread;


    public static void main(String[] args) throws Exception {
        // initializes variables
        init();

        // set up main thread by getting output file
        String outputFileDir = getOutputFileDir();
        mainThread = new Thread (() -> {
            try {
                mainThreadFunction(outputFileDir);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        // if a file directory is chosen then begin sensing and recording
        if (outputFileDir != null) mainThread.start();
        else System.exit(0);
    }

    // Initializes variables
    static void init() throws FrameGrabber.Exception {
        // Get webcam
        grabber = new OpenCVFrameGrabber(0);
        grabber.start();
        // init converter
        converter = new OpenCVFrameConverter.ToIplImage();
        // get initial webcam frame
        live = converter.convert(grabber.grab());
        // Frame that displays the current frame
        liveFrame = new CanvasFrame("Live Cam");
        liveFrame.setCanvasSize(live.width(), live.height());
        // adds a handler for closing the frame
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            mainThread.interrupt();
            exitAction();
        }));
        liveFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

    // Handles all webcam grabbing, motion sensing, and recording
    static void mainThreadFunction(String outputFileDir) throws Exception {
        IplImage diff = IplImage.create(live.width(), live.height(), IPL_DEPTH_8U, 1);
        IplImage image = null;
        IplImage prevImage = null;
        // Used to store contours
        CvMemStorage storage = CvMemStorage.create();

        // Each iteration grabs a new frame from webcam
        while (!Thread.currentThread().isInterrupted() && grabber != null && (live = converter.convert(grabber.grab())) != null) {
            cvClearMemStorage(storage);

            // Smooths using gaussian blur (removes much background noise)
            cvSmooth(live, live, CV_GAUSSIAN, 9, 9, 2, 2);

            // creates image and sets prev image
            if (image != null) prevImage = image;
            image = IplImage.create(live.width(), live.height(), IPL_DEPTH_8U, 1);
            cvCvtColor(live, image, CV_RGB2GRAY);

            // if previous image exists, determine the difference between them for motion
            if (prevImage != null) {
                // difference between curr and prev frame (motion sense)
                cvAbsDiff(image, prevImage, diff);
                // do some threshold for wipe away useless details
                cvThreshold(diff, diff, 30, 255, CV_THRESH_BINARY);

                // find largest bounding rectangles for live canvas
                int[][] bp = findLargestBoundingRect(diff, storage);
                // draw the largest bounding motion rectangle
                drawBoundingRect(live, bp);

                // if there is motion then start/continue recording
                if (bp[0][0] != -1) {
                    if (!recording && !Thread.currentThread().isInterrupted()) {
                        startRecording(live, outputFileDir);
                    }
                    framesWithoutMotion = 0;
                }
                // if there is no motion for a set amount of frames, stop recording
                else {
                    ++framesWithoutMotion;
                    if (framesWithoutMotion > MAXFRAMESWITHOUTMOTION && !Thread.currentThread().isInterrupted()) {
                        stopRecording();
                    }
                }
            }

            // display the live camera
            // add a timestamp to live frame
            cvPutText(live, new Date().toString(), cvPoint(5, live.height() - 5), cvFont(1, 1), CvScalar.BLACK);

            if (recording && !Thread.currentThread().isInterrupted()) {
                recorder.record(converter.convert(live));

            }

            if (recording && !Thread.currentThread().isInterrupted()) {
                cvPutText(live, "Rec", cvPoint(5, 15), cvFont(1, 1), CvScalar.RED);
            }

            // put live cam image onto 2nd canvas containing largest bounding motion rectangle
            liveFrame.showImage(converter.convert(live));
        }

    }


    // Uses jfilechooser to get a file directory to store videos
    static String getOutputFileDir() {
        // Choose directory to store video files
        JFileChooser chooser;
        chooser = new JFileChooser();
        chooser.setCurrentDirectory(new java.io.File("."));
        chooser.setDialogTitle("Select directory to store video files.");
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        chooser.setAcceptAllFileFilterUsed(false);
        if (chooser.showOpenDialog(liveFrame) == JFileChooser.APPROVE_OPTION) {
            String dirName = chooser.getSelectedFile().toString();
            dirName = dirName.replace("\\", "/");
            return dirName;
        }
        return null;
    }

    static void startRecording(IplImage image, String outputFileDir) throws FrameRecorder.Exception {
        // recorder
        String format = "mp4";
        String outputFile = outputFileDir + "/" + DATE_FORMAT.format(new Date()) + "." + format;
        System.out.println(outputFile);
        recorder = FrameRecorder.createDefault(outputFile, image.width(), image.height());
        recorder.setVideoCodec(avcodec.AV_CODEC_ID_H264);
        recorder.setFormat(format);
        recorder.start();
        recording = true;
    }

    static void stopRecording() throws FrameRecorder.Exception {
        recording = false;
        recorder.stop();
        recorder.release();
    }

    // what the app does when it is exiting
    static void exitAction() {
        try {
            if (recording) stopRecording();
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            grabber.stop();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // returns top left, top right, bottom left, bottom right points
    static int[][] findLargestBoundingRect(IplImage diff, CvMemStorage storage) {
        // initialize as all -1
        int[][] res = new int[4][2];
        for (int i = 0; i < res.length; ++i) {
            for (int j= 0; j < res[0].length; ++j) {
                res[i][j] = -1;
            }
        }

        /** Helpers **/
        // recognize contours
        CvSeq contour = new CvSeq(null);
        cvFindContours(diff, storage, contour, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        // for each contour find the largest bounding box
        for (; contour != null && !contour.isNull() && contour.elem_size() > 0; contour = contour.h_next()) {
            if (contour.elem_size() <= 0) continue;
            CvBox2D box = cvMinAreaRect2(contour, storage);
            if (box == null) continue;

            CvPoint2D32f center = box.center();
            CvSize2D32f size = box.size();

            int smallX = (int) (center.x() - size.width());
            int largeX = (int) (center.x() + size.width());
            int smallY = (int) (center.y() - size.height());
            int largeY = (int) (center.y() +  size.height());

            if (res[0][0] == -1) {
                res[0] = new int[] {smallX, smallY};
                res[1] = new int[] {largeX, smallY};
                res[2] = new int[] {smallX, largeY};
                res[3] = new int[] {largeX, largeY};
                continue;
            }

            if (smallX < res[0][0]) {
                res[0][0] = smallX;
                res[2][0] = smallX;
            }
            if (smallY < res[0][1]) {
                res[0][1] = (smallY);
                res[1][1] = (smallY);
            }
            if (largeX > res[1][0]) {
                res[1][0] = largeX;
                res[3][0] = largeX;
            }
            if (largeY > res[2][1]) {
                res[2][1] = (largeY);
                res[3][1] = (largeY);
            }
        }
        return res;
    }

    // draws the bounding rect given 2d array corresponding with points
    static void drawBoundingRect(IplImage image, int[][] points) {
        if (points[0][0] != -1) {
            cvLine(image, points[0], points[1], CvScalar.RED);
            cvLine(image, points[0], points[2], CvScalar.RED);
            cvLine(image, points[3], points[1], CvScalar.RED);
            cvLine(image, points[3], points[2], CvScalar.RED);
        }
    }
}