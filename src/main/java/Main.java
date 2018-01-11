import org.bytedeco.javacpp.ARToolKitPlus.*;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacv.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;


public class Main {
    public static void main(String[] args) throws FrameGrabber.Exception {
        FrameGrabber grabber = FrameGrabber.createDefault(0);
        grabber.start();


        Frame grabbedImage;
        CanvasFrame cFrame = new CanvasFrame("title", CanvasFrame.getDefaultGamma() / grabber.getGamma());
        while ((grabbedImage = grabber.grab()) != null) {
            OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
            Mat mat = converter.convert(grabbedImage);
            medianBlur(mat, mat, 15);
            cFrame.showImage(converter.convert(mat));
        }
    }
}