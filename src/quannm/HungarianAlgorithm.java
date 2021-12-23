package quannm;

import org.opencv.core.Mat;

import java.util.ArrayList;

public class HungarianAlgorithm {
    private static final double DBL_EPSILON = 2.22044604925031308084726333618164062e-16;;

    public HungarianAlgorithm() {
    }

    public Double Solve(Mat DistMatrix, ArrayList<Integer> Assignment){
//        System.out.println("solve");
        int nRows = (int) DistMatrix.size().height;
        int nCols = (int) DistMatrix.size().width;
//        System.out.println("hung rowcol "+ nRows+" "+nCols);
        double[] distMatrixIn = new double[nRows*nCols];
        int[] assignment = new int[nRows];
        double cost = 0.0;
        for ( int i = 0; i < nRows; i++)
            for ( int j = 0; j < nCols; j++)
                distMatrixIn[i + nRows * j] = DistMatrix.get(i,j)[0];

        assignmentoptimal(assignment, cost, distMatrixIn, nRows, nCols);

        Assignment.clear();
        for (int r = 0; r < nRows; r++){
//            System.out.println(assignment[r]);
            Assignment.add(assignment[r]);
        }

        return cost;
    }

    public void assignmentoptimal(int [] assignment, double cost, double[] distMatrixIn, int nOfRows, int nOfColumns){
        /* initialization */
//        System.out.println("assignment optimal");
        cost = 0;

        for (int row = 0 ; row < nOfRows; row++)
            assignment[row] = -1;
        /* generate working copy of distance Matrix */
        /* check if all matrix elements are positive */
        int nOfElements = nOfRows * nOfColumns;
        double[] distMatrix = new double[nOfElements];
        for (int row = 0 ; row < nOfElements; row++){
            double value = distMatrixIn[row];
            assert value >= 0 : "All matrix elements have to be non-negative.\n";

            distMatrix[row] = value;
        }

        boolean[] coveredColumns = new boolean[nOfColumns];
        boolean[] coveredRows = new boolean[nOfRows];
        boolean[] starMatrix = new boolean[nOfElements];
        boolean[] primeMatrix = new boolean[nOfElements];
        boolean[] newStarMatrix = new boolean[nOfElements];
        int minDim;
        /* preliminary steps */

        if (nOfRows <= nOfColumns){
            minDim = nOfRows;
            double minValue,value;

            for (int row = 0; row < nOfRows; row++){
                int current_index = row;
                minValue = distMatrix[current_index];
                current_index += nOfRows;
                while(current_index<nOfElements){
                    value = distMatrix[current_index];
                    if(value<minValue){
                        minValue = value;
                    }
                    current_index += nOfRows;
                }

                current_index = row;
                while(current_index<nOfElements){
                    distMatrix[current_index] -= minValue;
                    current_index += nOfRows;
                }
            }
            /* Steps 1 and 2a */
            for(int row =0 ; row<nOfRows;row++){
                for(int col =0 ; col<nOfColumns;col++){
                    if(Math.abs(distMatrix[row+nOfRows*col])<DBL_EPSILON){
                        if(!coveredColumns[col]){
                            starMatrix[row+nOfRows*col] = true;
                            coveredColumns[col] = true;
                            break;
                        }
                    }
                }
            }
        }
        else /* if(nOfRows > nOfColumns) */
        {
            minDim = nOfColumns;
            double value;
            int current_index;
            for (int col =0 ; col < nOfColumns; col++){
                /* find the smallest element in the column */
                current_index =  nOfRows*col;
                int columnEnd =  nOfRows*col + nOfRows;
                double minValue = distMatrix[current_index];
                while (current_index<columnEnd){
                    value = distMatrix[current_index];
                    current_index++;
                    if (value < minValue)
                        minValue = value;
                }
                /* subtract the smallest element from each element of the column */
                current_index =  nOfRows*col;
                while(current_index<columnEnd){
                    distMatrix[current_index] -= minValue;
                    current_index++;
                }
            }
            /* Steps 1 and 2a */
            for(int col = 0; col< nOfColumns;col++){
                for (int row = 0 ; row <nOfRows; row++){
                    if(Math.abs(distMatrix[row+nOfRows*col])<DBL_EPSILON){
                        if(!coveredRows[row]){
                            starMatrix[row+ nOfRows*col] = true;
                            coveredColumns[col] = true;
                            coveredRows[row] = true;
                            break;
                        }
                    }
                }
            }
            for (int row = 0; row < nOfRows; row++)
                coveredRows[row] = false;
        }

        /* move to step 2b */
        step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
        /* compute cost and remove invalid assignments */

        computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);
        return;
    }
    void buildassignmentvector(int[] assignment, boolean[] starMatrix, int nOfRows, int nOfColumns){
//        System.out.println("buildassignmentvector");
        int row,col;
        for (row = 0; row < nOfRows;row++)
            for (col = 0; col <nOfColumns; col++)
                if(starMatrix[row+ nOfRows*col]){
                    assignment[row] = col;
                    break;
                }
    }

    void computeassignmentcost(int[] assignment, double cost, double[] distMatrix, int nOfRows){
//        System.out.println("computeassignmentcost");
        int row,col;
        for (row = 0 ;row< nOfRows; row++){
            col = assignment[row];
            if(col>=0)
                cost+= distMatrix[row+nOfRows*col];
        }
    }

    void step2a(int[] assignment, double[] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix, boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns, int minDim ){
//        System.out.println("step2a");
        int columnEnd;
        int col;
        /* cover every column containing a starred zero */
        for (col = 0; col<nOfColumns; col++)
        {
            int currentIndex =  nOfRows*col;
            columnEnd =  nOfRows*col + nOfRows;
            while (currentIndex < columnEnd){
                if (starMatrix[currentIndex])
                {
                    coveredColumns[col] = true;
                    break;
                }
                currentIndex++;
            }
        }
//        System.out.println("step2a1");
        /* move to step 3 */
        step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }

    void step2b(int[] assignment, double[] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix, boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns, int minDim ){
//        System.out.println("step2b");
        int col, nOfCoveredColumns;

        /* count covered columns */
        nOfCoveredColumns = 0;
        for (col = 0; col<nOfColumns; col++)
            if (coveredColumns[col])
                nOfCoveredColumns++;

        if (nOfCoveredColumns == minDim)
        {
            /* algorithm finished */
            buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
        }
        else
        {
            /* move to step 3 */

            step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
        }
    }

    void step3(int[] assignment, double[] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix, boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns, int minDim ){
//        System.out.println("step3");
        boolean zerosFound;
        int row,col,starCol;
        zerosFound = true;
        while(zerosFound){
            zerosFound = false;
            for (col = 0; col<nOfColumns; col++)
                if (!coveredColumns[col])
                    for (row = 0; row<nOfRows; row++)
                        if ((!coveredRows[row]) && (Math.abs(distMatrix[row + nOfRows*col]) < DBL_EPSILON))
                        {
                            /* prime zero */
                            primeMatrix[row + nOfRows*col] = true;

                            /* find starred zero in current row */
                            for (starCol = 0; starCol<nOfColumns; starCol++)
                                if (starMatrix[row + nOfRows*starCol])
                                    break;

                            if (starCol == nOfColumns) /* no starred zero found */
                            {
                                /* move to step 4 */
                                step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                                return;
                            }
                            else
                            {
                                coveredRows[row] = true;
                                coveredColumns[starCol] = false;
                                zerosFound = true;
                                break;
                            }
                        }
        }
        /* move to step 5 */

        step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }

    void step4(int[] assignment, double[] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix, boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col ){
//        System.out.println("step4");
        int n, starRow, starCol, primeRow, primeCol;
        int nOfElements = nOfRows*nOfColumns;

        /* generate temporary copy of starMatrix */
        for (n = 0; n<nOfElements; n++)
            newStarMatrix[n] = starMatrix[n];

        /* star current zero */
        newStarMatrix[row + nOfRows*col] = true;

        /* find starred zero in current column */
        starCol = col;
        for (starRow = 0; starRow<nOfRows; starRow++)
            if (starMatrix[starRow + nOfRows*starCol])
                break;

        while (starRow<nOfRows)
        {
            /* unstar the starred zero */
            newStarMatrix[starRow + nOfRows*starCol] = false;

            /* find primed zero in current row */
            primeRow = starRow;
            for (primeCol = 0; primeCol<nOfColumns; primeCol++)
                if (primeMatrix[primeRow + nOfRows*primeCol])
                    break;

            /* star the primed zero */
            newStarMatrix[primeRow + nOfRows*primeCol] = true;

            /* find starred zero in current column */
            starCol = primeCol;
            for (starRow = 0; starRow<nOfRows; starRow++)
                if (starMatrix[starRow + nOfRows*starCol])
                    break;
        }

        /* use temporary copy as new starMatrix */
        /* delete all primes, uncover all rows */
        for (n = 0; n<nOfElements; n++)
        {
            primeMatrix[n] = false;
            starMatrix[n] = newStarMatrix[n];
        }
        for (n = 0; n<nOfRows; n++)
            coveredRows[n] = false;

        /* move to step 2a */
        step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
    void step5(int[] assignment, double[] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix, boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns, int minDim ){
//        System.out.println("step5");
        double h, value;
        int row, col;
        double DBL_MAX = (double) 1.79769313486231570814527423731704357e+308;
        /* find smallest uncovered element h */
        h = DBL_MAX;
        for (row = 0; row<nOfRows; row++)
            if (!coveredRows[row])
                for (col = 0; col<nOfColumns; col++)
                    if (!coveredColumns[col])
                    {
                        value = distMatrix[row + nOfRows*col];
                        if (value < h)
                            h = value;
                    }

        /* add h to each covered row */
        for (row = 0; row<nOfRows; row++)
            if (coveredRows[row])
                for (col = 0; col<nOfColumns; col++)
                    distMatrix[row + nOfRows*col] += h;

        /* subtract h from each uncovered column */
        for (col = 0; col<nOfColumns; col++)
            if (!coveredColumns[col])
                for (row = 0; row<nOfRows; row++)
                    distMatrix[row + nOfRows*col] -= h;
        /* move to step 3 */
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }

}
