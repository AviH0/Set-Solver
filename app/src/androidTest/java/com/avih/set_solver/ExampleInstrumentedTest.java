package com.avih.set_solver;

import android.content.Context;

import com.avih.set_solver.image_processing.ImageProcessor;
import com.avih.set_solver.set_game.SetCard;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.TensorFlowLite;

import static com.avih.set_solver.image_processing.ImageProcessor.getCardContours;
import static com.avih.set_solver.image_processing.ImageProcessor.unWarpCard;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.junit.Assert.*;



/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class ExampleInstrumentedTest {


    private final String TEST_IMAGE_PATH_1 = "C:\\Users\\Avinoam\\AndroidStudioProjects\\set_solver\\app\\src\\test\\res\\test_image.jpg";
    private final String TEST_IMAGE_PATH_2 = "C:\\Users\\Avinoam\\AndroidStudioProjects\\set_solver\\app\\src\\test\\res\\test_image_2.jpg";

    private Context c;

    @Before
    public void useAppContext() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertEquals("com.avih.set_solver", appContext.getPackageName());
        c = appContext;
    }

    @Test
    public void testDiamondDetection()
    {
        TensorFlowLite.init();
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