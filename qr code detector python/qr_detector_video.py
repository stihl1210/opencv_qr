import cv2
import numpy as np


CV_QR_NORTH = 0;
CV_QR_EAST = 1;
CV_QR_SOUTH = 2;
CV_QR_WEST = 3;

def showImg(name,frame):
    if(frame is not None):
        cv2.imshow(name,frame)

def calDist(P, Q):
    return np.sqrt(pow(abs(P[0] - Q[0]),2) + pow(abs(P[1] - Q[1]),2)) ;

def ptToLine_prev(lptA, lptB, ptC):
    ac = np.subtract(ptC, lptA)
    ab = np.subtract(lptB, lptA);

    prod = np.cross(ac,ab);

    distAB = calDist(lptA, lptB)

    return prod/distAB;

def ptToLine(lptA, lptB, ptc):


    try:
        a = -((lptB[1] - lptA[1]) / (lptB[0] - lptA[0]));
        b = 1.0;
        c = (((lptB[1] - lptA[1]) /(lptB[0] - lptA[0])) * lptA[0]) - lptA[1];
        if(len(ptc) == 2):
            pdist = (a * ptc[0] + (b * ptc[1]) + c) / np.sqrt((a * a) + (b * b));
        else:
            pdist = 0;


    except ZeroDivisionError:
        print lptA
        print lptB
        print ptc
        print "zeroDivisionError"
        return 0

    return pdist;

def calSlope(ptA, ptB):
    dP = np.subtract(ptA, ptB)

    if(dP[1]==0):
        return (0.0, 0);
    else:
        return (dP[1]/dP[0], 1);

def getVertices(contours, c_id, slope):
    x,y,w,h = cv2.boundingRect(contours[c_id]);
    tl = (x,y)
    br = (x+w,y+h)

    M0=M1=M2=M3 = list([0,0])
    A = B = C=D=W=X=Y=Z = list([0,0])

    A = tl
    B[0] = br[0]
    B[1] = tl[1]

    C = br
    D[0] =tl[0]
    D[1] = br[1]

    W[0] = (A[0] + B[0]) / 2;
    W[1] = A[1];

    X[0] = B[0];
    X[1] = (B[1] + C[1]) / 2;

    Y[0] = (C[0] + D[0]) / 2;
    Y[1] = C[1];

    Z[0] = D[0];
    Z[1] = (D[1] + A[1]) / 2;

    dmax = list()
    dmax.append(0.0);
    dmax.append(0.0);
    dmax.append(0.0);
    dmax.append(0.0);

    pd1=pd2=0.0

    if (slope > 5 or slope < -5 ):

        for i in range(len(contours[c_id])):

            pd1 = ptToLine(C,A,contours[c_id][i]);
            pd2 = ptToLine(B,D,contours[c_id][i]);

            if((pd1 >= 0.0) and (pd2 > 0.0)):
                dmax[1],M1 =updateCorner(contours[c_id][i], W ,dmax[1],M1);

            elif((pd1 > 0.0) and (pd2 <= 0.0)):
                dmax[2],M2 =updateCorner(contours[c_id][i], X ,dmax[2],M2);
            elif((pd1 <= 0.0) and (pd2 < 0.0)):
                dmax[3],M3 =updateCorner(contours[c_id][i], Y ,dmax[3],M3);
            elif((pd1 < 0.0) and (pd2 >= 0.0)):
                dmax[0],M0 =updateCorner(contours[c_id][i], Z ,dmax[0],M0);
            else:
                continue;

    else:
        halfx = (A[0] + B[0]) / 2;
        halfy = (A[1] + D[1]) / 2;

        for i in range(len(contours[c_id])):

            if((contours[c_id][i][0][0] < halfx) and (contours[c_id][i][0][1] <= halfy)):
                dmax[2], M0 =updateCorner(contours[c_id][i][0],C,dmax[2],M0);
            elif((contours[c_id][i][0][0] >= halfx) and (contours[c_id][i][0][1] < halfy)):
                dmax[3],M1 =updateCorner(contours[c_id][i][0],D,dmax[3],M1);
            elif((contours[c_id][i][0][0] > halfx) and (contours[c_id][i][0][1] >= halfy)):
                dmax[0],M2 =updateCorner(contours[c_id][i][0],A,dmax[0],M2);
            elif((contours[c_id][i][0][0] <= halfx) and (contours[c_id][i][0][1] > halfy)):
                dmax[1],M3 =updateCorner(contours[c_id][i][0],B,dmax[1],M3);

    ret = list()
    ret.append(M0)
    ret.append(M1)
    ret.append(M2)
    ret.append(M3)
    return ret;

def updateCorner(point, ref, baseline, corner):
    temp_dist = calDist(point,ref);

    if(temp_dist > baseline):
        baseline = temp_dist;
        return (baseline, point);
    else:
        return (baseline, corner);


def updateCornerOr(orientation, IN):

    M0=M1=M2=M3 = 0
    if(orientation == CV_QR_NORTH):
        M0 = IN[0];
        M1 = IN[1];
        M2 = IN[2];
        M3 = IN[3];

    elif(orientation == CV_QR_EAST):
        M0 = IN[1];
        M1 = IN[2];
        M2 = IN[3];
        M3 = IN[0];
    elif(orientation == CV_QR_SOUTH):
        M0 = IN[2];
        M1 = IN[3];
        M2 = IN[0];
        M3 = IN[1];
    elif(orientation == CV_QR_WEST):
        M0 = IN[3];
        M1 = IN[0];
        M2 = IN[1];
        M3 = IN[2];

    out = (M0,M1,M2,M3)

    return out

def getIntersectionPoint(a1,a2,b1,b2):
    p = 0
    q = 0
    r = 0
    s = 0
    try:
        p = a1;
        q = b1;
        r = np.array(a2)-np.array(a1)
        s = np.array(b2)-np.array(b1)
        print p
        print q
        print r
        print s


        if(cross(r,s) == 0):
            return (0.0)

        t = cross(q-p,s)/cross(r,s);

        print t

        intersection = p*1.0 + t*r*1.0;
        return intersection
    except Exception as e:
        print 'except - getInterSectionPoint'
        print e
        raise
        print p
        print q
        print r
        print s

        return np.array([0,0])


def cross(v1,v2):
    return 1.0*v1[0]*v2[1] - 1.0*v1[1]*v2[0];


if __name__ == "__main__":

    cap = cv2.VideoCapture(0);
    while(True):

        ret, img = cap.read();

        import copy

        traces = copy.copy(img);
        qr_thres = copy.copy(img);
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        edges = cv2.Canny(gray, 100, 200, 3);
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]

        mark = 0;
        mu = list()
        mc = list()



        for i in range(len(contours)):
            mu.append(cv2.moments(contours[i], False))
            if(mu[i]['m00'] != 0.0):
                mc.append((mu[i]['m10']/mu[i]['m00'], mu[i]['m01']/mu[i]['m00']))
            else:
                mc.append([0,0]);

        A = 0
        B = 0
        C = 0

        for i in range(len(contours)):
            k = i
            c = 0
            while(hierarchy[k][2] != -1):
                k = hierarchy[k][2]
                c = c+1
            if(hierarchy[k][2] != -1):
                c=c+1

            if(c>=5):
                if( mark == 0):
                    A = i;
                elif(mark == 1):
                    B= i;
                elif(mark == 2):
                    C = i

                mark = mark +1;

        median1 = 0;
        median2 = 0;
        outlier = 0;

        if( mark >= 2):

            AB = calDist(mc[A],mc[B]);
            BC = calDist(mc[B],mc[C]);
            CA = calDist(mc[C],mc[A]);

            if ( AB > BC and AB > CA ):
                outlier = C; median1=A; median2=B;
            elif( CA > AB and CA > BC ):
                outlier = B; median1=A; median2=C;
            elif (BC > AB and BC > CA ):
                outlier = A;  median1=B; median2=C;

            top = outlier

            align = 0;

            dist = ptToLine(mc[median1], mc[median2], mc[outlier]);
            slope, align = calSlope(mc[median1], mc[median2]);
            right=bottom =  0;
            orientation = -1;

            if(align == 0):
                bottom = median1
                right = median2;
            elif(slope <0 and dist < 0):
                bottom = median1
                right = median2;
                orientation = CV_QR_NORTH
            elif(slope > 0 and dist < 0):
                right = median1
                bottom = median2;
                orientation = CV_QR_EAST
            elif(slope <0 and dist > 0):
                right = median1
                bottom = median2;
                orientation = CV_QR_SOUTH
            elif(slope > 0 and dist > 0):
                bottom = median1
                right = median2;
                orientation = CV_QR_WEST

            areaTop = areaRight = areaBottom = 0;
            qr_raw = np.zeros((100,100));
            qr = np.zeros((100,100));
            qr_thres = np.zeros((100,100));

            if(top < len(contours) and right< len(contours) and bottom< len(contours) and cv2.contourArea(contours[top]) > 10 and cv2.contourArea(contours[right]) >10 and cv2.contourArea(contours[bottom])>10):
                tempL = tempM = tempO = N = L = M = O = [0,0]
                warpMat = list()

                tempL = getVertices(contours, top, slope);
                tempM = getVertices(contours, right, slope);
                tempO = getVertices(contours, bottom, slope);

                L = updateCornerOr(orientation, tempL)
                M = updateCornerOr(orientation, tempM)
                O = updateCornerOr(orientation, tempO)


                N = getIntersectionPoint(M[1],M[2],O[3],O[2])
                lam = 0.5;
                N = (int(N[0] + lam),int(N[1]+ lam))

                src = np.array([L[0],M[1],N,O[3]],dtype=np.float32)
                dst = np.array([[0,0],[100,0],[100,100],[0,100]],dtype=np.float32)

                if(len(src) == 4 and len(dst) == 4 ):
                    warpMatrix = cv2.getPerspectiveTransform(src,dst);
                    qr_raw = cv2.warpPerspective(img, warpMatrix, (100,100))
                    qr = cv2.copyMakeBorder(qr_raw,  10,10,10,10,cv2.BORDER_CONSTANT, (255,255,255))
                    qr_gray = cv2.cvtColor(qr , cv2.COLOR_RGB2GRAY);
                    _ , qr_thres = cv2.threshold(qr_gray, 127, 255, cv2.THRESH_BINARY)

                cv2.drawContours(img, contours, top, (255,200,0), 2, 8 , hierarchy, 0);
                cv2.drawContours(img, contours, right, (0,0,255), 2, 8 , hierarchy, 0);
                cv2.drawContours(img, contours, bottom, (255,0,100), 2, 8 , hierarchy, 0);

                DBG = 1
                if(DBG==1):

                    if(slope > 5):
                        cv2.circle(traces, (10,20), 5, (0,0,255), -1, 8, 0);
                    elif(slope < -5):
                        cv2.circle(traces, (10,20), 5, (255,255,255), -1, 8, 0);

                    cv2.drawContours(traces, contours, top, (255,0,100), 1, 8, hierarchy, 0);
                    cv2.drawContours(traces, contours, right, (255,0,100), 1, 8, hierarchy, 0);
                    cv2.drawContours( traces, contours, bottom , (255,0,100), 1, 8, hierarchy, 0 );
                    cv2.circle( traces, tuple(L[0]), 2,  (255,255,0), -1, 8, 0 );
                    cv2.circle( traces, tuple(L[1]), 2,  (0,255,0), -1, 8, 0 );
                    cv2.circle( traces, tuple(L[2]), 2,  (0,0,255), -1, 8, 0 );
                    cv2.circle( traces, tuple(L[3]), 2,  (128,128,128), -1, 8, 0 );

                    cv2.circle( traces, tuple(M[0]), 2,  (255,255,0), -1, 8, 0 );
                    cv2.circle( traces, tuple(M[1]), 2,  (0,255,0), -1, 8, 0 );
                    cv2.circle( traces, tuple(M[2]), 2,  (0,0,255), -1, 8, 0 );
                    cv2.circle( traces, tuple(M[3]), 2,  (128,128,128), -1, 8, 0 );

                    cv2.circle( traces, tuple(O[0]), 2,  (255,255,0), -1, 8, 0 );
                    cv2.circle( traces, tuple(O[1]), 2,  (0,255,0), -1, 8, 0 );
                    cv2.circle( traces, tuple(O[2]), 2,  (0,0,255), -1, 8, 0 );
                    cv2.circle( traces, tuple(O[3]), 2,  (128,128,128), -1, 8, 0 );

                    cv2.circle( traces, tuple(N), 2,  (255,255,255), -1, 8, 0 );
                    cv2.line(traces, tuple(M[1]), tuple(N), (0, 0, 255), 1, 8, 0);
                    cv2.line(traces,tuple(O[3]), tuple(N), (0,0,255),1,8,0);

                    fontFace = cv2.FONT_HERSHEY_PLAIN;

                    if(orientation == CV_QR_NORTH):
                        cv2.putText(traces, "NORTH", (20,30), fontFace, 1, (0, 255, 0), 1, 8);
                    elif (orientation == CV_QR_EAST):
                        cv2.putText(traces, "EAST", (20,30), fontFace, 1, (0, 255, 0), 1, 8);
                    elif (orientation == CV_QR_SOUTH):
                        cv2.putText(traces, "SOUTH", (20,30), fontFace, 1, (0, 255, 0), 1, 8);
                    elif(orientation == CV_QR_WEST):
                        cv2.putText(traces, "WEST", (20,30), fontFace, 1, (0, 255, 0), 1, 8);



        showImg("basic",img)
        showImg("traces",traces)
        showImg("qr_threes", qr_thres);

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;

    cap.release()
    cv2.destroyAllWindows()

