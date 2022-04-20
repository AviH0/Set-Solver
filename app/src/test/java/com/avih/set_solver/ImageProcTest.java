package com.avih.set_solver;

import android.app.Application;
import android.app.Instrumentation;

import com.avih.set_solver.image_processing.CardProcessor;
import com.avih.set_solver.image_processing.ImageProcessor;
import com.avih.set_solver.set_game.SetCard;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.RuntimeEnvironment;
import org.tensorflow.lite.TensorFlowLite;

import static com.avih.set_solver.image_processing.ImageProcessor.getCardContours;
import static com.avih.set_solver.image_processing.ImageProcessor.procImage;
import static com.avih.set_solver.image_processing.ImageProcessor.unWarpCard;
import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

import android.content.Context;


@RunWith(RobolectricTestRunner.class)
public class ImageProcTest {

    private final String OUTPUT_PATH = "C:\\Users\\Avinoam\\AndroidStudioProjects\\set_solver\\app\\src\\test\\res\\out\\";
    private final String TEST_IMAGE_PATH_BASE = "C:\\Users\\Avinoam\\AndroidStudioProjects\\set_solver\\app\\src\\test\\res\\test_image_";
    private final String TEST_IMAGE_PATH_1 = "C:\\Users\\Avinoam\\AndroidStudioProjects\\set_solver\\app\\src\\test\\res\\test_image.jpg";
    private final String TEST_IMAGE_PATH_2 = "C:\\Users\\Avinoam\\AndroidStudioProjects\\set_solver\\app\\src\\test\\res\\test_image_2.jpg";

    @Test
    public void testExtractCard_1()
    {
        Mat src = imread(TEST_IMAGE_PATH_1);
        // Check if everything was fine
        if (src.data().isNull())
            return;
        // Show source image
        MatVector cardContours = getCardContours(src);
        for (int i = 0; i < cardContours.size(); i++)
        {
            Mat card = unWarpCard(src, cardContours, i);
            imwrite(TEST_IMAGE_PATH_1 + ".card" + i + ".jpg", card);
        }
    }

    @Test
    public void testCardProcessor()
    {
        Context c = RuntimeEnvironment.getApplication();
        CardProcessor cardProcessor = new CardProcessor(c);

    }

    @Test
    public void testExtractCard_all()
    {
        Context c = RuntimeEnvironment.getApplication();
        ImageProcessor imageProcessor = new ImageProcessor(c, R.raw.diamond, R.raw.round, R.raw.squiggle);
        for (int x = 0; x < 5; x++) {
            String image_path = TEST_IMAGE_PATH_BASE + (x+1) + ".jpg";
            Mat src = imread(image_path);
            // Check if everything was fine
            if (src.data().isNull())
                return;
            // Show source image
            MatVector cardContours = getCardContours(src);
            for (int i = 0; i < cardContours.size(); i++) {
                Mat card = unWarpCard(src, cardContours, i);
                boolean isCardValue = imageProcessor.isCard(card);
//                System.out.println("img " + (x+1) + ", card " + i + " card value: " + isCardValue);
//                if (isCardValue)
                imwrite(OUTPUT_PATH + "img" + (x+1) + "_card_" + i + ".jpg", card);
            }
        }
    }



    @Test
    public void testExtractCard_2()
    {

        Mat src = imread(TEST_IMAGE_PATH_2);
        // Check if everything was fine
        if (src.data().isNull())
            return;
        // Show source image
        MatVector cardContours = getCardContours(src);

        for (int i = 0; i < cardContours.size(); i++)
        {
            Mat card = unWarpCard(src, cardContours, i);
            imwrite(TEST_IMAGE_PATH_2 + ".card" + i + ".jpg",  card);
        }
    }
    @Test
    public void testDiamondDetection()
    {
        TensorFlowLite.init();
        Context c = RuntimeEnvironment.getApplication();
        Mat src = imread(TEST_IMAGE_PATH_2);
        // Check if everything was fine
        if (src.data().isNull())
            return;
        // Show source image
        MatVector cardContours = getCardContours(src);

        for (int i = 0; i < cardContours.size(); i++)
        {
            Mat card = unWarpCard(src, cardContours, i);
            ImageProcessor imageProcessor = new ImageProcessor(c, R.raw.diamond, R.raw.round, R.raw.squiggle);
            SetCard.Shape shape = imageProcessor.getCardShape(card);
            System.out.println("Card " + i + " is: " + shape);
//            imwrite(TEST_IMAGE_PATH_2 + ".card" + i + "diamond" + isDiamond +   ".jpg",  card);
        }
    }

}
