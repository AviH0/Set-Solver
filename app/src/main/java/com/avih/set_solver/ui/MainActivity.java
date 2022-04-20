package com.avih.set_solver.ui;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.camera2.interop.Camera2CameraInfo;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.impl.SurfaceConfig;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;


import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.hardware.camera2.CameraCharacteristics;
import android.os.Bundle;
import android.util.Size;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import com.avih.set_solver.R;
import com.avih.set_solver.image_processing.CardProcessor;
import com.google.common.util.concurrent.ListenableFuture;


import org.opencv.osgi.OpenCVNativeLoader;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

interface PreviewOverlay
{
    void overlay(Bitmap bmp);
}

public class MainActivity extends AppCompatActivity implements PreviewOverlay {


    private static final int REQUEST_CODE = 999;
    private static final String[] REQUIRED_PERMISSIONS = {Manifest.permission.CAMERA};
    private Executor cameraExecutor;
    private ImageView overlayView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        overlayView = findViewById(R.id.overlay_image);
        new OpenCVNativeLoader().init();


        cameraExecutor = Executors.newSingleThreadExecutor();
        if(requiredPermissionsGranted())
        {
            startCamera();
        }
        else{
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE);
        }

    }

    private boolean requiredPermissionsGranted()
    {
        for (String permission :
                REQUIRED_PERMISSIONS) {

            if (checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED)
            {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE) {
            if (requiredPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(
                        this,
                        getString(R.string.permission_deny_text),
                        Toast.LENGTH_SHORT
                ).show();
                finish();
            }
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {



                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    Preview preview = new Preview.Builder().build();
//                new CameraSelector.Builder().
                    PreviewView viewFinder = findViewById(R.id.view_finder);


                    ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                            .setOutputImageRotationEnabled(false)
                            .setTargetResolution(new Size(viewFinder.getWidth()/4, viewFinder.getHeight()/4))
//                            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                            .build();
                    imageAnalysis.setAnalyzer(cameraExecutor, new Analyzer(MainActivity.this, MainActivity.this));
                    cameraProvider.unbindAll();
                    Camera camera = cameraProvider.bindToLifecycle(MainActivity.this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis);

                    preview.setSurfaceProvider(viewFinder.getSurfaceProvider());
                } catch (ExecutionException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }



            }
        }, ContextCompat.getMainExecutor(this));
    }

    @Override
    public void overlay(Bitmap bmp) {
        Paint drawPaint = new Paint();
        drawPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.ADD));
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                overlayView.setLayerPaint(drawPaint);
                overlayView.setImageBitmap(bmp);
            }
        });

    }

    private class Analyzer implements ImageAnalysis.Analyzer {
        private CardProcessor cardProc;
        private PreviewOverlay previewOverlay;

        public Analyzer(Context c, PreviewOverlay callback) {
            cardProc = new CardProcessor(c);
            previewOverlay = callback;
        }

        @SuppressLint("UnsafeOptInUsageError")
        @Override
        public void analyze(@NonNull ImageProxy image) {
            Bitmap overlay = cardProc.processImage(image.getImage(), new Size(overlayView.getWidth(), overlayView.getHeight()));
            previewOverlay.overlay(overlay);
            image.close();
        }

    }


}