/*
 * Built upon Angelo Marchesin sample motion detector on javaCV sample files.
 */

import org.bytedeco.javacpp.*;
import org.bytedeco.javacv.*;

import javax.swing.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class MotionDetector {
    public static void main(String[] args) throws Exception {
        OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
        OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
        grabber.start();

        IplImage live = converter.convert(grabber.grab());
        IplImage image = null;
        IplImage prevImage = null;
        IplImage diff = null;

        // Frame that displays the difference between curr and prev frames
        CanvasFrame diffFrame = new CanvasFrame("Difference in previous frame");
        diffFrame.setCanvasSize(live.width(), live.height());

        // Frame that displays the current frame
        CanvasFrame liveFrame = new CanvasFrame("Live Cam");
        liveFrame.setCanvasSize(live.width(), live.height());

        diffFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        liveFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        // Used to store contours
        CvMemStorage storage = CvMemStorage.create();

        while (diffFrame.isVisible() && liveFrame.isVisible() && (live = converter.convert(grabber.grab())) != null) {
            cvClearMemStorage(storage);

            // Smooths using gaussian blur
            cvSmooth(live, live, CV_GAUSSIAN, 9, 9, 2, 2);

            // creates image if not created and sets prev image
            if (image == null) {
                image = IplImage.create(live.width(), live.height(), IPL_DEPTH_8U, 1);
                cvCvtColor(live, image, CV_RGB2GRAY);
            } else {
                prevImage = image;
                image = IplImage.create(live.width(), live.height(), IPL_DEPTH_8U, 1);
                cvCvtColor(live, image, CV_RGB2GRAY);
            }

            // sets differential image if not set
            if (diff == null) {
                diff = IplImage.create(live.width(), live.height(), IPL_DEPTH_8U, 1);
            }

            if (prevImage != null) {
                // difference between curr and prev frame (motion sense)
                cvAbsDiff(image, prevImage, diff);
                // do some threshold for wipe away useless details
                cvThreshold(diff, diff, 30, 255, CV_THRESH_BINARY);

                // find largest bounding rectangles for live canvas
                int[][] bp = findLargestBoundingRect(diff, storage);
                // draw the largest bounding motion rectangle
                if (bp[0][0] != -1) {
                    cvLine(live, bp[0], bp[1], CvScalar.RED);
                    cvLine(live, bp[0], bp[2], CvScalar.RED);
                    cvLine(live, bp[3], bp[1], CvScalar.RED);
                    cvLine(live, bp[3], bp[2], CvScalar.RED);
                }
            }
            // put live cam image onto 2nd canvas containing largest bounding motion rectangle
            liveFrame.showImage(converter.convert(live));
            // put the motion sensed frame onto canvas
            if (prevImage != null) diffFrame.showImage(converter.convert(diff));
        }
        grabber.stop();
        diffFrame.dispose();
        liveFrame.dispose();
    }

    // Returns top left, top right, bottom left, bottom right points
    static int[][] findLargestBoundingRect(IplImage diff, CvMemStorage storage) {
        // initialize as all -1
        int[][] res = new int[4][2];
        for (int i = 0; i < res.length; ++i) {
            for (int j= 0; j < res[0].length; ++j) {
                res[i][j] = -1;
            }
        }

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
}