package com.avih.set_solver.image_processing;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ColorSpace;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsic3DLUT;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
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
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_core.RotatedRect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.opencv.android.Utils;
import org.opencv.utils.Converters;
import org.tensorflow.lite.support.image.TensorImage;

import java.nio.ByteBuffer;

import static com.avih.set_solver.image_processing.ImageProcessor.getCardContours;
import static com.avih.set_solver.image_processing.ImageProcessor.getContourRect;
import static com.avih.set_solver.image_processing.ImageProcessor.unWarpCard;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2BGRA;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_4;
import static org.bytedeco.opencv.global.opencv_imgproc.boxPoints;
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
        Bitmap bitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
        YuvToRgbConverter yuvToRgbConverter = new YuvToRgbConverter(context);
        yuvToRgbConverter.yuvToRgb(image, bitmap);
        Mat src = imageProcessor.bitmapToMat(bitmap);
//        resize(src, src, new org.bytedeco.opencv.opencv_core.Size( targetResolution.getWidth(), targetResolution.getHeight()));

        // Check if everything was fine
        if (src.data().isNull())
            return null;
        // Show source image
        Mat overlay = Mat.zeros(src.size(), opencv_core.CV_8UC4).asMat();
        cvtColor(overlay, overlay, opencv_imgproc.COLOR_RGB2RGBA);
        imwrite(context.getCacheDir().getAbsolutePath() + "/src_image.jpg", src);
        MatVector cardContours = getCardContours(src);
        for (int i = 0; i < cardContours.size(); i++) {
            if (imageProcessor.isCard(cardContours.get(i).clone())) {
                Mat card = unWarpCard(src, cardContours, i);
                SetCard.Shape cardShape = imageProcessor.getCardShape(card);
//                imwrite(context.getCacheDir().getAbsolutePath() + "/card" + i + ".jpg", card);
                if (cardShape != null) {
                    SetCard.Color cardColor = imageProcessor.getCardColor(card);
                    SetCard.Number cardNumber = imageProcessor.getCardNumber(card);
                    SetCard setCard = new SetCard(cardShape, cardColor, cardNumber);
                    imwrite(context.getCacheDir().getAbsolutePath() + "/card" + i + setCard + ".jpg", card);
                    Mat rectPoints = new Mat();
                    boxPoints(getContourRect(cardContours, i), rectPoints);
                    Indexer indexer = rectPoints.createIndexer();
//                    opencv_imgproc.rectangle(overlay,
//                            new Point((int) indexer.getDouble(1, 0),(int)  indexer.getDouble(1, 1)),
//                            new Point((int) indexer.getDouble(2, 0),(int)  indexer.getDouble(2, 1)),
//                            Scalar.ALPHA255, 5, LINE_4, 0);
                    opencv_imgproc.rectangle(overlay,
                            new Point((int) indexer.getDouble(0, 0),(int)  indexer.getDouble(0, 1)),
                            new Point((int) indexer.getDouble(2, 0),(int)  indexer.getDouble(2, 1)),
                            Scalar.ALPHA255, 5, LINE_4, 0);
//                    drawContours(overlay, new MatVector(new Mat[]{rectPoints}), 0, Scalar.ALPHA255, 5, LINE_4, null, 0, null);
                    Log.i(TAG, "processImage: Found shape: " + cardShape.name());
                }
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
        BitmapFactory.Options options = new BitmapFactory.Options();
        Bitmap bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
//        RenderScript rs = RenderScript.create(context);
//        ScriptIntrinsicYuvToRGB intrinsicYuvToRGB = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));
//        Type.Builder builder = new Type.Builder(rs, Element.U8(rs)).setX(bitmap.getWidth()).setY(bitmap.getHeight());
//        Allocation inData = Allocation.createTyped(rs, builder.create(), Allocation.USAGE_SCRIPT);
//        inData.copyFrom(bitmap);
//
//        Type.Builder argbType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(bitmap.getWidth()).setY(bitmap.getHeight());
//        Allocation outData = Allocation.createTyped(rs, argbType.create(), Allocation.USAGE_SCRIPT);
//
//        intrinsicYuvToRGB.setInput(inData);
//        intrinsicYuvToRGB.forEach(outData);
//        Bitmap resBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
//        outData.copyTo(resBitmap);
        return bitmap;
    }
}
