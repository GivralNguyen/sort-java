package quannm;

import org.opencv.core.Rect;

public class TrackingBox {
    private int frame;
    private int id;
    private Rect box;

    public int getFrame() {
        return frame;
    }

    public void setFrame(int frame) {
        this.frame = frame;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public Rect getBox() {
        return box;
    }

    public void setBox(Rect box) {
        this.box = box;
    }

}
