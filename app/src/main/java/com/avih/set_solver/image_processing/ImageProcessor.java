package com.avih.set_solver.image_processing;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;

import com.avih.set_solver.R;
import com.avih.set_solver.ml.ShapeModel;
import com.avih.set_solver.set_game.SetCard;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;

import org.bytedeco.librealsense.context;
import org.bytedeco.opencv.opencv_core.*;

import org.opencv.android.Utils;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;


import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ImageProcessor {

    private Mat diamond_mat, squiggle_mat, round_mat, cardHist;
    private Context _context;
    private ShapeModel model;

    public ImageProcessor(Context c, int diamond_id, int round_id, int squiggle_id)
    {
        _context = c;
        diamond_mat = readImageFromResources(c, diamond_id);
        MatVector contours = getCardContours(diamond_mat);
        diamond_mat = Mat.zeros(diamond_mat.size(), CV_8U).asMat();
        drawContours(diamond_mat, contours, 0, Scalar.WHITE);
        squiggle_mat = readImageFromResources(c, squiggle_id);
        round_mat = readImageFromResources(c, round_id);

        FloatPointer histRange = new FloatPointer(0f, 255f);
        cardHist = calcMatHist(squiggle_mat);
        try {
            model = ShapeModel.newInstance(_context);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (UnsatisfiedLinkError e)
        {
            e.printStackTrace();
        }
    }


    private static final int[] WHITE = {255, 255, 255};
    private static final int[] BLACK = {0, 0, 0};

    public SetCard.Shape getCardShape(@ByRef Mat img)
    {
            if (model == null){
                return null;
            }
            img.convertTo(img, CV_8U);
            // Creates inputs for reference.
//            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);
            TensorImage input = TensorImage.fromBitmap(matToBitmap(img));

//            TensorBuffer floatBuffer = TensorBufferFloat.createFrom(input.getTensorBuffer(), DataType.FLOAT32);
//            input.load(floatBuffer);
//            inputFeature0.loadBuffer(img.asByteBuffer());

            // Runs model inference and gets result.
            ShapeModel.Outputs outputs = model.process(input.getTensorBuffer());
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();



            float[] pred = outputFeature0.getFloatArray();
            if (pred[0] > 150)
            {
                return SetCard.Shape.DIAMOND;
            }
            if (pred[1] > 150)
            {
                return SetCard.Shape.SQUIGGLE;
            }
            if (pred[2] > 150)
            {
                return SetCard.Shape.CIRCLE;
            }
            return null;

    }

    private Mat calcMatHist(Mat src)
    {
        Mat hist = new Mat();
        FloatPointer histRange = new FloatPointer(0f, 255f);
        int histBins = 150;
        cvtColor(src, src, COLOR_RGB2GRAY);
        calcHist(src,
                1,
                new IntPointer(0,0),
                new Mat(),
                hist,
                1,
                new IntPointer(histBins, histBins),
                new PointerPointer<FloatPointer>(histRange, histRange),
                true,
                false);
        return hist;
    }

    public boolean isCard(Mat card)
    {
        Mat hist = calcMatHist(card.clone());
        return compareHist(hist, cardHist, HISTCMP_CHISQR) >= 20_000;
    }

    public static Mat getContourRect(MatVector contours, int index)
    {
        Mat approx = new Mat(CV_32F);
        approxPolyDP(contours.get(index), approx, 0.01 * arcLength(contours.get(index), true), true);
        //calculate the center of mass of our contour image using moments
        Moments moment = moments(approx);
        int x = (int) (moment.m10() / moment.m00());
        int y = (int) (moment.m01() / moment.m00());

//SORT POINTS RELATIVE TO CENTER OF MASS

        Mat sortedSrcPoints = new Mat(new Size(2, 4), CV_8U);
        Indexer indexer = approx.createIndexer();
        Indexer sortedSrcPointsIndexer = sortedSrcPoints.createIndexer();
        for(int i=0; i<approx.rows(); i++){
            int datax = (int) indexer.getDouble(0, i, 0);
            int datay = (int) indexer.getDouble(0, i, 1);
            if(datax < x && datay < y){
                sortedSrcPointsIndexer.putDouble(new long[]{0, 0}, datax);
                sortedSrcPointsIndexer.putDouble(new long[]{0, 1}, datay);
            }else if(datax > x && datay < y){
                sortedSrcPointsIndexer.putDouble(new long[]{1, 0}, datax);
                sortedSrcPointsIndexer.putDouble(new long[]{1, 1}, datay);
            }else if (datax < x && datay > y){
                sortedSrcPointsIndexer.putDouble(new long[]{2, 0}, datax);
                sortedSrcPointsIndexer.putDouble(new long[]{2, 1}, datay);
            }else if (datax > x && datay > y){
                sortedSrcPointsIndexer.putDouble(new long[]{3, 0}, datax);
                sortedSrcPointsIndexer.putDouble(new long[]{3, 1}, datay);
            }
        }
        double height = Math.abs(sortedSrcPointsIndexer.getDouble(0, 1) - sortedSrcPointsIndexer.getDouble(2, 1));
        double width = Math.abs(sortedSrcPointsIndexer.getDouble(0, 0) - sortedSrcPointsIndexer.getDouble(1, 0));
//        if (height > width)
//        {
//            double datax = sortedSrcPointsIndexer.getDouble(3, 0);
//            double datay = sortedSrcPointsIndexer.getDouble(3, 1);
////            sortedSrcPointsIndexer.putDouble(new long[]{3, 0}, sortedSrcPointsIndexer.getDouble(0, 0));
////            sortedSrcPointsIndexer.putDouble(new long[]{3, 1}, sortedSrcPointsIndexer.getDouble(0, 1));
////
////            sortedSrcPointsIndexer.putDouble(new long[]{0, 0}, datax);
////            sortedSrcPointsIndexer.putDouble(new long[]{0, 1}, datay);
//
//            datax = sortedSrcPointsIndexer.getDouble(2, 0);
//            datay = sortedSrcPointsIndexer.getDouble(2, 1);
//            sortedSrcPointsIndexer.putDouble(new long[]{2, 0}, sortedSrcPointsIndexer.getDouble(1, 0));
//            sortedSrcPointsIndexer.putDouble(new long[]{2, 1}, sortedSrcPointsIndexer.getDouble(1, 1));
//
//            sortedSrcPointsIndexer.putDouble(new long[]{1, 0}, datax);
//            sortedSrcPointsIndexer.putDouble(new long[]{1, 1}, datay);
//        }

        return sortedSrcPoints;
    }

    public static Mat unWarpCard(@ByRef Mat src, @ByRef MatVector contours, int index)
    {
//        Mat contour = Mat.zeros(src.size(0), src.size(1), CV_8U).asMat();
//        drawContours(contour, contours, index, Scalar.WHITE);
//        Rect boundingRect = boundingRect(contour);
//        contour = contour.colRange(
//                boundingRect.x(),  boundingRect.x() + boundingRect.width()).rowRange(boundingRect.y(), boundingRect.y() + boundingRect.height());
//        src = src.colRange(
//                boundingRect.x(), boundingRect.x() +  boundingRect.width()).rowRange(boundingRect.y(), boundingRect.y() + boundingRect.height());


        Mat rect = new Mat(new Size(2, 4), CV_32F, new FloatPointer(0, 0, 224, 0, 0, 100, 224, 100));

        Mat approx = new Mat(CV_32F);
        approxPolyDP(contours.get(index), approx, 0.01 * arcLength(contours.get(index), true), true);
        //calculate the center of mass of our contour image using moments
        Moments moment = moments(approx);
        int x = (int) (moment.m10() / moment.m00());
        int y = (int) (moment.m01() / moment.m00());

//SORT POINTS RELATIVE TO CENTER OF MASS

        Mat sortedSrcPoints = new Mat(new Size(2, 4), CV_32F);
        Indexer indexer = approx.createIndexer();
        Indexer sortedSrcPointsIndexer = sortedSrcPoints.createIndexer();
        for(int i=0; i<approx.rows(); i++){
            int datax = (int) indexer.getDouble(0, i, 0);
            int datay = (int) indexer.getDouble(0, i, 1);
            if(datax < x && datay < y){
                sortedSrcPointsIndexer.putDouble(new long[]{0, 0}, datax);
                sortedSrcPointsIndexer.putDouble(new long[]{0, 1}, datay);
            }else if(datax > x && datay < y){
                sortedSrcPointsIndexer.putDouble(new long[]{1, 0}, datax);
                sortedSrcPointsIndexer.putDouble(new long[]{1, 1}, datay);
            }else if (datax < x && datay > y){
                sortedSrcPointsIndexer.putDouble(new long[]{2, 0}, datax);
                sortedSrcPointsIndexer.putDouble(new long[]{2, 1}, datay);
            }else if (datax > x && datay > y){
                sortedSrcPointsIndexer.putDouble(new long[]{3, 0}, datax);
                sortedSrcPointsIndexer.putDouble(new long[]{3, 1}, datay);
            }
        }
        double height = Math.abs(sortedSrcPointsIndexer.getDouble(0, 1) - sortedSrcPointsIndexer.getDouble(2, 1));
        double width = Math.abs(sortedSrcPointsIndexer.getDouble(0, 0) - sortedSrcPointsIndexer.getDouble(1, 0));
        if (height > width)
        {
            double datax = sortedSrcPointsIndexer.getDouble(3, 0);
            double datay = sortedSrcPointsIndexer.getDouble(3, 1);
//            sortedSrcPointsIndexer.putDouble(new long[]{3, 0}, sortedSrcPointsIndexer.getDouble(0, 0));
//            sortedSrcPointsIndexer.putDouble(new long[]{3, 1}, sortedSrcPointsIndexer.getDouble(0, 1));
//
//            sortedSrcPointsIndexer.putDouble(new long[]{0, 0}, datax);
//            sortedSrcPointsIndexer.putDouble(new long[]{0, 1}, datay);

            datax = sortedSrcPointsIndexer.getDouble(2, 0);
            datay = sortedSrcPointsIndexer.getDouble(2, 1);
            sortedSrcPointsIndexer.putDouble(new long[]{2, 0}, sortedSrcPointsIndexer.getDouble(1, 0));
            sortedSrcPointsIndexer.putDouble(new long[]{2, 1}, sortedSrcPointsIndexer.getDouble(1, 1));

            sortedSrcPointsIndexer.putDouble(new long[]{1, 0}, datax);
            sortedSrcPointsIndexer.putDouble(new long[]{1, 1}, datay);
        }

        Mat perspectiveTransform = getPerspectiveTransform(sortedSrcPoints, rect);
        Mat dst = Mat.zeros(src.size(), src.type()).asMat();
        warpPerspective(src, dst, perspectiveTransform, new Size(224, 100));
//        resize(dst, dst, new Size(200, 200));
        Mat final_result = new Mat(new Size(224, 224));
        copyMakeBorder(dst, final_result, 62, 62, 0, 0, BORDER_CONSTANT, Scalar.all(0));
        return final_result;
    }

    public static Mat procImage(Mat src) {

        MatVector contours = getCardContours(src);
        src.convertTo(src, CV_8UC3);
        // Create the marker image for the watershed algorithm
        Mat markers = Mat.zeros(src.size(), CV_32SC1).asMat();
        // Draw the foreground markers
        Mat test = Mat.zeros(src.size(), CV_8UC1).asMat();
        for (int i = 0; i < contours.size(); i++)
            drawContours(markers, contours, i, Scalar.all((i) + 1));
//        if (true)
//        return test;
        // Draw the background marker
        circle(markers, new Point(5, 5), 3, RGB(255, 255, 255));
//        imshow("Markers", multiply(markers, 10000).asMat());

        // Perform the watershed algorithm
        watershed(src, markers);
        if (true)
            return unWarpCard(src, contours, 3);


        Mat mark = Mat.zeros(markers.size(), CV_8UC1).asMat();
        markers.convertTo(mark, CV_8UC1);
        bitwise_not(mark, mark);
//            imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
        // image looks like at that point
        // Generate random colors
        List<int[]> colors = new ArrayList<int[]>();
        for (int i = 0; i < contours.size(); i++) {
            int b = theRNG().uniform(0, 255);
            int g = theRNG().uniform(0, 255);
            int r = theRNG().uniform(0, 255);
            int[] color = { b, g, r };
            colors.add(color);
        }
        // Create the result image
        Mat dst = Mat.zeros(markers.size(), CV_8UC3).asMat();
        // Fill labeled objects with random colors
        IntIndexer markersIndexer = markers.createIndexer();
        UByteIndexer dstIndexer = dst.createIndexer();
        for (int i = 0; i < markersIndexer.size(0); i++) {
            for (int j = 0; j < markersIndexer.size(1); j++) {
                int index = markersIndexer.get(i, j);
                if (index > 0 && index <= contours.size())
                    dstIndexer.put(i, j, colors.get(index - 1));
                else
                    dstIndexer.put(i, j, BLACK);
            }
        }
        // Visualize the final image
//        imshow("Final Result", dst);
        return markers;
    }


    public static MatVector getCardContours(Mat src)
    {
        // Change the background from white to black, since that will help later to extract
        // better results during the use of Distance Transform
        src.convertTo(src, CV_8U);
        UByteIndexer srcIndexer = src.createIndexer();
        for (int x = 0; x < srcIndexer.size(0); x++) {
            for (int y = 0; y < srcIndexer.size(1); y++) {
                int[] values = new int[3];
                srcIndexer.get(x, y, values);
                if (Arrays.equals(values, WHITE)) {
                    srcIndexer.put(x, y, BLACK);
                }
            }
        }
        // Show output image
//        imshow("Black Background Image", src);

        // Create a kernel that we will use for accuting/sharpening our image
        Mat kernel = Mat.ones(3, 3, CV_32F).asMat();
        FloatIndexer kernelIndexer = kernel.createIndexer();
        kernelIndexer.put(1, 1, -8); // an approximation of second derivative, a quite strong kernel

        // do the laplacian filtering as it is
        // well, we need to convert everything in something more deeper then CV_8U
        // because the kernel has some negative values,
        // and we can expect in general to have a Laplacian image with negative values
        // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
        // so the possible negative number will be truncated
        Mat imgLaplacian = new Mat();
        Mat sharp = src; // copy source image to another temporary one
        filter2D(sharp, imgLaplacian, CV_32F, kernel);
        src.convertTo(sharp, CV_32F);
        Mat imgResult = subtract(sharp, imgLaplacian).asMat();
        // convert back to 8bits gray scale
        imgResult.convertTo(imgResult, CV_8UC3);
        imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
        // imshow( "Laplace Filtered Image", imgLaplacian );
//        imshow("New Sharped Image", imgResult);

        src = imgResult; // copy back
        // Create binary image from source image
        Mat bw = new Mat();
        cvtColor(src, bw, CV_BGR2GRAY);
        threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
//        imshow("Binary Image", bw);

        // Perform the distance transform algorithm
        Mat dist = new Mat(bw);
//        distanceTransform(bw, dist, CV_DIST_L2, 3);
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        normalize(dist, dist, 0, 1., NORM_MINMAX, -1, null);
//        imshow("Distance Transform Image", dist);

        // Threshold to obtain the peaks
        // This will be the markers for the foreground objects
        threshold(dist, dist, .1, 1., CV_THRESH_BINARY);
        // Dilate a bit the dist image
        Mat kernel1 = Mat.ones(3, 3, CV_8UC1).asMat();
        dilate(dist, dist, kernel1);
//        imshow("Peaks", dist);
        // Create the CV_8U version of the distance image
        // It is needed for findContours()
        Mat dist_8u = new Mat();
        dist.convertTo(dist_8u, CV_8U);
        // Find total markers
        MatVector contours = new MatVector();
        findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        return contours;
    }

    public Mat bitmapToMat(Bitmap bmp)
    {
        Mat empty_mat = new Mat(bmp.getHeight(), bmp.getWidth(), CV_8U);
        empty_mat.deallocate(false);
        org.opencv.core.Mat src_bad = new org.opencv.core.Mat(empty_mat.address());
        Utils.bitmapToMat(bmp, src_bad);

        Mat src =  new Mat((Pointer)null) { { address = src_bad.getNativeObjAddr(); } };
        Mat ret = new Mat(src);
        assert ret.address() != src.address();
        src.deallocate(false);
//        ret.deallocate(false);
        return ret;
    }

    public Bitmap matToBitmap(Mat mat)
    {
        Mat copy = mat.clone();
        copy.deallocate(false);
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_4444);
        org.opencv.core.Mat src = new org.opencv.core.Mat(copy.address());
        Utils.matToBitmap(src, bitmap);

        return bitmap;
    }
    public Mat readImageFromResources(Context c, int id)
    {
        String filename = "tempfile.jpg";
        File file = new File(c.getCacheDir() + "/" + filename);
        if (!file.exists())
            try {

                InputStream is = c.getResources().openRawResource(id);
                int size = is.available();
                byte[] buffer = new byte[size];
                is.read(buffer);
                is.close();

                FileOutputStream fos = new FileOutputStream(file);

                fos.write(buffer);
                fos.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        if (file.exists()) {
            return imread(file.getAbsolutePath());
        }
        return null;
    }
}
