package com.avih.set_solver.ui;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

public class OverlayView extends View
{
    private Bitmap bmp, oldBmp;
    Paint drawPaint, erasePaint;
    int clearSave;
    boolean isSaved = false;
    public OverlayView(Context context) {
        super(context);
    }

    public OverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public OverlayView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    public OverlayView(Context context, @Nullable AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
    }

    public void setBitmap(Bitmap bmp)
    {
        oldBmp = this.bmp;
        this.bmp = bmp;
        erasePaint = new Paint();
        erasePaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST));
        drawPaint = new Paint();
        drawPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.ADD));

    }

    @Override
    protected void onDraw(Canvas canvas) {
        if (bmp == null)
        {
            return;
        }
        if (oldBmp != null)
        {

        }
        canvas.drawBitmap(bmp, 0, 0, drawPaint);
    }
}