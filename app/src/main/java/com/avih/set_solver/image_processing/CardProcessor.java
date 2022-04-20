package com.avih.set_solver.image_processing;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;
import android.util.Size;
import android.widget.ImageView;
import android.widget.Toast;

import com.avih.set_solver.R;
import com.avih.set_solver.set_game.SetCard;
import com.google.firebase.crashlytics.buildtools.reloc.org.apache.commons.io.output.ByteArrayOutputStream;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.librealsense.context;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.opencv.android.Utils;
import org.opencv.utils.Converters;
import org.tensorflow.lite.support.image.TensorImage;

import java.nio.ByteBuffer;

import static com.avih.set_solver.image_processing.ImageProcessor.getCardContours;
import static com.avih.set_solver.image_processing.ImageProcessor.unWarpCard;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2BGRA;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_4;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.drawContours;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class CardProcessor {

    private static final String TAG = "CP";
    private ImageProcessor imageProcessor;
    private Context context;

    public CardProcessor(Context c) {
        context = c;
        imageProcessor = new ImageProcessor(c, R.raw.diamond, R.raw.round, R.raw.squiggle);
    }

    public Bitmap processImage(Image image, Size targetResolution) {
        Bitmap bitmap = toBitmap(image);

        Mat src = imageProcessor.bitmapToMat(bitmap);
//        resize(src, src, new org.bytedeco.opencv.opencv_core.Size( targetResolution.getWidth(), targetResolution.getHeight()));

        // Check if everything was fine
        if (src.data().isNull())
            return null;
        // Show source image
        Mat overlay = Mat.zeros(src.size(), opencv_core.CV_8UC4).asMat();
        cvtColor(overlay, overlay, opencv_imgproc.COLOR_RGB2RGBA);
        MatVector cardContours = getCardContours(src);
        for (int i = 0; i < cardContours.size(); i++) {
            Mat card = unWarpCard(src, cardContours, i);
            SetCard.Shape cardShape = imageProcessor.getCardShape(card);
            if (cardShape != null) {
                imwrite(context.getCacheDir().getAbsolutePath() + "/card" + i + ".jpg", card);
//                Indexer indexer = ImageProcessor.getContourRect(cardContours, i).createIndexer();
//                opencv_imgproc.rectangle(overlay, new org.bytedeco.opencv.opencv_core.Rect(
//                                new Point((int) indexer.getDouble(1, 0),
//                                        (int) indexer.getDouble(1, 1)),
//                                new Point((int) indexer.getDouble(3, 0),
//                                        (int) indexer.getDouble(3, 1))),
//                        Scalar.ALPHA255, 5, LINE_4, 0);
                drawContours(overlay, cardContours, i, Scalar.ALPHA255, 5, LINE_4, null, 0, null);
                Log.i(TAG, "processImage: Found shape: " + cardShape.name());
            }
        }
        resize(overlay, overlay, new org.bytedeco.opencv.opencv_core.Size(targetResolution.getWidth(), targetResolution.getHeight()));
        return imageProcessor.matToBitmap(overlay);
    }

    private Bitmap toBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }
}
