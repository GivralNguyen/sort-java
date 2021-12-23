package quannm;

import org.opencv.core.*;
import org.opencv.video.KalmanFilter;

import java.util.ArrayList;

import static org.opencv.core.CvType.CV_32F;

public class KalmanTracker {

    public static int kf_count = 0;
    public int m_time_since_update;
    public int m_hits;

    public static int getKf_count() {
        return kf_count;
    }

    public static void setKf_count(int kf_count) {
        KalmanTracker.kf_count = kf_count;
    }

    public int m_hit_streak;
    public int m_age;
    public int m_id;

    private KalmanFilter kf = new KalmanFilter();
    private Mat measurement;
    private ArrayList<Rect> m_history = new ArrayList<>();

    public KalmanTracker(Rect initRect)
    {
        init_kf(initRect);
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
        kf_count++;
    }


    private void init_kf(Rect stateMat){
        int stateNum = 7;
        int measureNum = 4;
        kf = new KalmanFilter(stateNum, measureNum, 0,CvType.CV_32F);
        Mat transitionMatrix = new Mat(7,7,CvType.CV_32F,new Scalar(0));
        float[] tM  = {
                1,0,0,0,1,0,0,
                0,1,0,0,0,1,0,
                0,0,1,0,0,0,1,
                0,0,0,1,0,0,0,
                0,0,0,0,1,0,0,
                0,0,0,0,0,1,0,
                0,0,0,0,0,0,1};
        transitionMatrix.put(0,0,tM);
        kf.set_transitionMatrix(transitionMatrix);

        kf.set_measurementMatrix(Mat.eye(4,7,CV_32F));

        Mat processNoiseCov = Mat.eye(7, 7, CvType.CV_32F);
        processNoiseCov = processNoiseCov.mul(processNoiseCov, 1e-2);
        kf.set_processNoiseCov(processNoiseCov);

        Mat id1 = Mat.eye(4,4, CvType.CV_32F);
        id1 = id1.mul(id1,1e-1);
        kf.set_measurementNoiseCov(id1);

        Mat id2 = Mat.eye(7,7, CvType.CV_32F);
        //id2 = id2.mul(id2,0.1);
        kf.set_errorCovPost(id2);

//        System.out.println("stateMat "+stateMat);
//        System.out.println("stateMatx "+stateMat.x);
//        System.out.println("stateMaty "+stateMat.y);
//        System.out.println("stateMatwidth "+stateMat.width);
//        System.out.println("stateMatheight "+stateMat.height);
//        System.out.println("stateMatarea "+stateMat.area());
//        System.out.println("stateMatratio "+(float)stateMat.width / (float) stateMat.height);
        Mat statePost = new Mat(7, 1, CvType.CV_32F, new Scalar(0));
        statePost.put(0,0,stateMat.x+stateMat.width/2);
        statePost.put(1,0,stateMat.y+stateMat.height/2);
        statePost.put(2,0,stateMat.area());
        statePost.put(3,0,(float)stateMat.width/(float)stateMat.height);
        kf.set_statePost(statePost);

//        System.out.println("transitionmat "+ kf.get_transitionMatrix().size());
//        System.out.println("transitionmat\n "+ kf.get_transitionMatrix().dump());
//        System.out.println("measurementMatrix "+ kf.get_measurementMatrix().size());
//        System.out.println("measurementMatrix\n "+ kf.get_measurementMatrix().dump());
//        System.out.println("processNoiseCov "+ kf.get_processNoiseCov().size());
//        System.out.println("processNoiseCov\n "+ kf.get_processNoiseCov().dump());
//        System.out.println("measurementNoiseCov "+ kf.get_measurementNoiseCov().size());
//        System.out.println("measurementNoiseCov\n "+ kf.get_measurementNoiseCov().dump());
//        System.out.println("errorCovPost "+ kf.get_errorCovPost().size());
//        System.out.println("errorCovPost\n "+ kf.get_errorCovPost().dump());
//        System.out.println("statePre "+ kf.get_statePre().size());
//        System.out.println("statePre\n "+ kf.get_statePre().dump());
//        System.out.println("statePost "+ kf.get_statePost().size());
//        System.out.println("statePost\n "+ kf.get_statePost().dump());

    }

    Rect predict(){
        Mat p = kf.predict();
//        System.out.println(" p "+ p.dump());
        m_age += 1;
        if(m_time_since_update > 0)
            m_hit_streak = 0;
        m_time_since_update += 1;
        Rect predictBox = get_rect_xysr((float) p.get(0,0)[0],(float) p.get(1,0)[0],(float) p.get(2,0)[0],(float) p.get(3,0)[0]);
        m_history.add(predictBox);
        return m_history.get(m_history.size()-1);
    }

    void update(Rect stateMat){
        m_time_since_update = 0;
        m_history.clear();
        m_hits += 1;
        m_hit_streak += 1;

        //measurement
//        float[][] floatMeasurementArray = new float[][]{{stateMat.x+stateMat.width/2},{stateMat.y + stateMat.height/2},{(int) stateMat.area()},{(float)stateMat.width/(float)stateMat.height}};
//        measurement = setMatrix(4,0,new Mat(4,0,CvType.CV_32F),floatMeasurementArray);
        Mat measurement = new Mat(4, 1, CvType.CV_32F, new Scalar(0));
        measurement.put(0,0,stateMat.x+stateMat.width/2);
        measurement.put(1,0,stateMat.y+stateMat.height/2);
        measurement.put(2,0,stateMat.area());
        measurement.put(3,0,(float)stateMat.width/(float)stateMat.height);
        // update
        kf.correct(measurement);
    }

    public Rect get_state(){
        Mat s = kf.get_statePost();
        return get_rect_xysr((float) s.get(0,0)[0],(float) s.get(1,0)[0],(float) s.get(2,0)[0],(float) s.get(3,0)[0]);
    }

    private Mat setMatrix(int rowNum, int colNum, Mat tempMatrix, float[][] intArray) {

        for (int row = 0; row < rowNum; row++) {
            for (int col = 0; col < colNum; col++)
                tempMatrix.put(row, col, intArray[row][col]);
        }
        return tempMatrix;
    }

    private Rect get_rect_xysr(float cx,float cy, float s, float r){
        int w = (int) Math.sqrt(s*r);
        int h = (int) (s / w);
        int x = (int) (cx - w / 2);
        int y = (int) (cy - h / 2);

        if (x < 0 && cx > 0)
            x = 0;
        if (y < 0 && cy > 0)
            y = 0;

        return new Rect(x, y, w, h);
    }



}
