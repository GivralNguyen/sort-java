package quannm;

//import com.google.common.collect.Sets;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.highgui.HighGui.*;

import java.util.List;
import java.util.Set;
import java.util.HashSet;

public class Main {

    static int total_frames = 0;
    static double total_time = 0.0;


    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        ArrayList<String> Sequences = new ArrayList<>(Arrays.asList("PETS09-S2L1", "TUD-Campus", "TUD-Stadtmitte", "ETH-Bahnhof", "ETH-Sunnyday", "ETH-Pedcross2", "KITTI-13", "KITTI-17", "ADL-Rundle-6", "ADL-Rundle-8", "Venice-2"));
        // Loop through all datasets
        for (String sequence : Sequences) {
            TestSORT(sequence, false); // display variable
        }
    }

    public static void TestSORT(String seqName, boolean display) throws IOException {
        int currTrackerIndex;
        List randColor = new ArrayList();
        float r,g,b;
        Color randomColor;
        Random rand = new Random();
        for (int i = 0; i < 20; i++)
        {
            r = rand.nextFloat();
            g = rand.nextFloat();
            b = rand.nextFloat();
            randomColor = new Color (r, g, b);
            randColor.add(randomColor);
        }

        System.out.println("Processing " + seqName + " ....");
        String imgPath = "src/mot_benchmark/train/" + seqName + "/img1/";
        if (display) {
            if (!Files.exists(Path.of(imgPath))) {
                System.out.println("Image path not found");
                display = false;
            }
        }
        // 1 . Read detection file
        String detFileName = "src/data/" + seqName + "/det.txt";
        File detFile = new File(detFileName);

        String detLine;
        ArrayList<TrackingBox> detData = new ArrayList<>();
        char ch;
        float tpx, tpy, tpw, tph;

        try (LineIterator it = FileUtils.lineIterator(detFile, "UTF-8")) {
            while (it.hasNext()) {
                TrackingBox tb = new TrackingBox();
                String line = it.nextLine();
                String[] values = line.split(",");

                tb.setFrame(Integer.parseInt(values[0]));
                tb.setId(Integer.parseInt(values[1]));
                tb.setBox(new Rect((int) Float.parseFloat(values[2]), (int) Float.parseFloat(values[3]), (int) Float.parseFloat(values[4]), (int) Float.parseFloat(values[5])));
                detData.add(tb);
            }
        }
        // 2. group detData by frame
        int maxFrame = 0;
        for (TrackingBox tb : detData) {
            if (maxFrame < tb.getFrame())
                maxFrame = tb.getFrame();
        }

        ArrayList<ArrayList<TrackingBox>> detFrameData = new ArrayList<>();
        ArrayList<TrackingBox> tempVec = new ArrayList<>();
        for (int fi = 0; fi < maxFrame; fi++) {
            for (TrackingBox tb : detData) {
                if (tb.getFrame() == fi + 1) {
                    tempVec.add(tb);
                }
            }
            detFrameData.add((ArrayList<TrackingBox>) tempVec.clone());
            tempVec.clear();

        }

        // 3. update across frames
        int frame_count = 0;
        int max_age = 1;
        int min_hits = 3;
        double iouThreshold = 0.3;
        ArrayList<KalmanTracker> trackers = new ArrayList<>();
        KalmanTracker.setKf_count(0);
        // variables used in the for-loop
        ArrayList<Rect> predictedBoxes = new ArrayList<>();
        Mat iouMatrix;
        ArrayList<Integer> assignment = new ArrayList<>();
        Set<Integer> unmatchedDetections = new HashSet<>();
        Set<Integer> unmatchedTrajectories = new HashSet<>();
        Set<Integer> allItems = new HashSet<>();
        Set<Integer> matchedItems = new HashSet<>();
        ArrayList<Point> matchedPairs = new ArrayList<>();
        ArrayList<TrackingBox> frameTrackingResult = new ArrayList<>();
        int trkNum = 0;
        int detNum = 0;

        double cycle_time = 0.0;
        long start_time = 0;

        FileWriter fstream = new FileWriter("src/output/" + seqName + ".txt", false);

        // main loop
        for (int fi = 0; fi < maxFrame; fi++) {
            total_frames++;
            frame_count++;

//            start_time = getTickCount();

            if (trackers.size() == 0) // the first frame met
            {
                // initialize kalman trackers using first detections.
                for (int i = 0; i < detFrameData.get(fi).size(); i++) {
                    KalmanTracker trk = new KalmanTracker(detFrameData.get(fi).get(i).getBox());
                    trackers.add(trk);
                }

                // output the first frame detections
                for (int id = 0; id < detFrameData.get(fi).size(); id++) {
                    TrackingBox tb = detFrameData.get(fi).get(id);
                    fstream.write(tb.getFrame() + "," + (id + 1) + "," + tb.getBox().x + "," + tb.getBox().y + "," + tb.getBox().width + "," + tb.getBox().height + ",1,-1,-1,-1" + "\n");
                }
                continue;
            }

            // 3.1. get predicted locations from existing trackers.
            predictedBoxes.clear();

//            System.out.println("trackers size "+trackers.size());
            for (currTrackerIndex = 0; currTrackerIndex < trackers.size();){
                Rect pBox = trackers.get(currTrackerIndex).predict();
//                try
//                {
//                    Thread.sleep(1000); // Sleep for one second
//                }
//                catch (InterruptedException e)
//                {
//                    Thread.currentThread().interrupt();
//                }
//                System.out.println("current index "+currTrackerIndex);
//                System.out.println("predicted box "+pBox);
                if (pBox.x >=0 && pBox.y >= 0){
                    predictedBoxes.add(pBox);
                    currTrackerIndex++;

                }
                else{
                    trackers.remove(currTrackerIndex);
                }
            }

            // 3.2. associate detections to tracked object (both represented as bounding boxes)
            // dets : detFrameData[fi]

            trkNum = predictedBoxes.size();
            detNum = detFrameData.get(fi).size();

            iouMatrix = new Mat(trkNum,detNum,CV_64F);

            for (int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
            {
                for (int j = 0; j < detNum; j++)
                {
                    // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                    iouMatrix.put(i,j,1 - GetIOU(predictedBoxes.get(i), detFrameData.get(fi).get(j).getBox()));
                }
            }

            // solve the assignment problem using hungarian algorithm.
            // the resulting assignment is [track(prediction) : detection], with len=preNum
            HungarianAlgorithm HungAlgo = new HungarianAlgorithm();
            assignment.clear();

            HungAlgo.Solve(iouMatrix,assignment);

            // find matches, unmatched_detections and unmatched_predictions

            unmatchedTrajectories.clear();

//            System.out.println("unmatched det "+ unmatchedDetections);

//            System.out.println("unmatched det "+ unmatchedDetections.size());
            unmatchedDetections.clear();
            allItems.clear();
            matchedItems.clear();
            if (detNum > trkNum) //	there are unmatched detections
            {
                for ( int n = 0; n < detNum; n++)
                allItems.add(n);

                for ( int i = 0; i < trkNum; ++i)
                matchedItems.add(assignment.get(i));
                unmatchedDetections = new HashSet<>(allItems);
                unmatchedDetections.removeAll(matchedItems);

            }
            else if (detNum < trkNum) // there are unmatched trajectory/predictions
            {
                for (int i = 0; i < trkNum; ++i)
                    if (assignment.get(i) == -1) // unassigned label will be set as -1 in the assignment algorithm
                        unmatchedTrajectories.add(i);
            }


            // filter out matched with low IOU
            matchedPairs.clear();
            for (int i = 0; i < trkNum; ++i)
            {
                if (assignment.get(i) == -1) // pass over invalid values
                    continue;
                if (((double)1 - (iouMatrix.get(i,assignment.get(i)))[0]) < iouThreshold)
                {
                    unmatchedTrajectories.add(i);
                    unmatchedDetections.add(assignment.get(i));
                }
                else
                    matchedPairs.add(new Point(i, assignment.get(i)));
            }
            ///////////////////////////////////////
            // 3.3. updating trackers

            // update matched trackers with assigned detections.
            // each prediction is corresponding to a tracker
//            System.out.println("3.3");
            int detIdx, trkIdx;
            for ( int i = 0; i < matchedPairs.size(); i++)
            {
                trkIdx = (int) matchedPairs.get(i).x;
                detIdx = (int) matchedPairs.get(i).y;
                trackers.get(trkIdx).update(detFrameData.get(fi).get(detIdx).getBox());
            }

            // create and initialise new trackers for unmatched detections

            for (Integer umd : unmatchedDetections)
            {
//                System.out.println(detFrameData.get(fi).get(umd).getBox());
                KalmanTracker tracker = new KalmanTracker(detFrameData.get(fi).get(umd).getBox());
                trackers.add(tracker);
            }
            // get trackers' output
            frameTrackingResult.clear();


            for (currTrackerIndex = 0; currTrackerIndex < trackers.size();){
                if((trackers.get(currTrackerIndex).m_time_since_update < 1) && (trackers.get(currTrackerIndex).m_hit_streak >= min_hits || frame_count <= min_hits)){
                    TrackingBox res = new TrackingBox();
                    res.setBox(trackers.get(currTrackerIndex).get_state());
                    res.setId(trackers.get(currTrackerIndex).m_id+1);
                    res.setFrame(frame_count);
                    frameTrackingResult.add(res);
                }
                currTrackerIndex++;
                // remove dead tracklet
                if (currTrackerIndex != (trackers.size()) && (trackers.get(currTrackerIndex)).m_time_since_update > max_age)
                    trackers.remove(currTrackerIndex);
            }
            cycle_time = (double)(getTickCount() - start_time);
            total_time += cycle_time / getTickFrequency();

            for (TrackingBox tb : frameTrackingResult)
            {
                fstream.write(tb.getFrame() + "," + (tb.getId()) + "," + tb.getBox().x + "," + tb.getBox().y + "," + tb.getBox().width + "," + tb.getBox().height + ",1,-1,-1,-1" + "\n");
            }
            if(display){
                String oss = imgPath + String.format("%06d", fi+1)+".jpg";
                System.out.println(oss);
                Mat img = Imgcodecs.imread(oss);
                if (img.empty())
                    continue;
                for (TrackingBox tb : frameTrackingResult){
                    Color color = (Color) randColor.get(tb.getId() % 20);
//                    System.out.println(color);
                    Scalar opencv_color = new Scalar(color.getBlue(),color.getGreen(),color.getRed());
                    Imgproc.rectangle(img,tb.getBox(), opencv_color, 2, 8, 0);
                }
                imshow(seqName,img);
                waitKey(40);
            }
        }
        fstream.close();
        if (display)
            destroyAllWindows();
    }

    static double GetIOU(Rect bb_test, Rect bb_gt)
    {
        Double DBL_EPSILON = 2.22044604925031308084726333618164062e-16;
        float intersectionMinX = Math.max(bb_test.x, bb_gt.x);
        float intersectionMinY = Math.max(bb_test.y, bb_gt.y);
        float intersectionMaxX = Math.min(bb_test.x+ bb_test.width, bb_gt.x+bb_gt.width);
        float intersectionMaxY = Math.min(bb_test.y+ bb_test.height, bb_gt.y+bb_gt.height);
        float in = Math.max(intersectionMaxY - intersectionMinY, 0) *
                Math.max(intersectionMaxX - intersectionMinX, 0);
        float un = (float) (bb_test.area() + bb_gt.area() - in);

        if (un < DBL_EPSILON)
            return 0;

        return (double)(in / un);
    }
}



