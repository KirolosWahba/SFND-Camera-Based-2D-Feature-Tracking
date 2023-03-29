# SFND Camera Based 2D Feature Tracking Writeup

## Data Buffer
Deque is used for O(1) time complexity operations instead of erasing from a vector.

    dataBuffer.push_back(frame);
    if(dataBuffer.size() > dataBufferSize) 
	    dataBuffer.pop_front();

## Key Points Detectors
SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT  detectors are implemented and selectable by setting a string accordingly as shown below.

    if (detectorType.compare("SHITOMASI") == 0)
    {
        detKeypointsShiTomasi(keypoints, imgGray, bVis);
    }
    else if (detectorType.compare("HARRIS") == 0)
    {
        detKeypointsHarris(keypoints, imgGray, bVis);
    }
    else if (detectorType.compare("FAST") == 0)
    {
        detKeypointsFAST(keypoints, imgGray, bVis);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detKeypointsBRISK(keypoints, imgGray, bVis);
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detKeypointsORB(keypoints, imgGray, bVis);
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detKeypointsAKAZE(keypoints, imgGray, bVis);
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detKeypointsSIFT(keypoints, imgGray, bVis);
    }

Key Points are removed using using rect.contains method to focus only
on the preceding vehicle 
  
    for(auto keypoint : keypoints)
    {
        if(vehicleRect.contains(keypoint.pt)) 
            filteredKeypoints.push_back(keypoint);
    }

## Descriptors 
BRISK, BRIEF, ORB, FREAK, AKAZE and SIFT descriptors are implemented and selectable by setting a string accordingly.

    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling 
								   // the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    
## Descriptors Matching
Brute-force and FLANN are implemented and selectable using the respective strings as follows.

    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    if (matcherType.compare("MAT_BF") == 0)
    {
    
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        std::cout << "BF matching. cross-check=" << crossCheck << std::endl;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        {   
            // OpenCV bug workaround : convert binary descriptors to floating point...
            // due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
    
        matcher = cv::FlannBasedMatcher::create();
        std::cout << "FLANN matching" << std::endl;
    }

Nearest Neighbor (**NN**) and K Nearest Neighbors (**KNN**) with the descriptor distance ratio test set to 0.8 matching are implemented as shown below.

    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
    
        double t = (double)cv::getTickCount();
    
        // Finds the best match for each descriptor in desc1
        matcher->match(descSource, descRef, matches);
    
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << std::endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
    
        std::vector<std::vector<cv::DMatch>> matchesPair;
        double t = (double)cv::getTickCount();
    
        matcher->knnMatch(descSource, descRef, matchesPair, 2);
    
        // Filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for( auto match : matchesPair ){
            if(match.at(0).distance/match.at(1).distance <= minDescDistRatio) 
                matches.emplace_back(match.at(0));
        }
    
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        std::cout << " (KNN & Distance Ratio Filtering) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << std::endl;
        std::cout << "# keypoints removed = " << matchesPair.size() - matches.size() << std::endl;          
    }

## Statistical Analysis 

Using `generateReport.py` file to automate all the possible combination of detector and descriptor pairs and using KNN approach for keypoints matching with (k=2) and distance ratio filtering = 0.8.

### Performance Evaluation 1
Counting the number of keypoints on the preceding vehicle detected by different detectors for all 10 images and taking a note of the distribution of their neighborhood size.

| Detector | img0 | img1 | img2 | img3 | img4 | img5 | img6 | img7 | img8 | img9 | Total Keypoints | Neighborhood size |
| :---:    | :---:  | :---:  | :---:  |  :---: | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---:  | :---: |
| SHI-TOMASI | 125 | 118 | 123 | 120 | 120 | 113 | 114 | 123 | 111 | 112 | 1179 | 4
| HARRIS | 17 | 14 | 18 | 21 | 26 | 43 | 18 | 30 | 26 | 34 | 247 | 6
| FAST | 419 | 427 | 404 | 423 | 386 | 414 | 418 | 406 | 396 | 401 | **4094** | 7
| BRISK | 264 | 282 | 282 | 277 | 297 | 279 | 289 | 272 | 266 | 254 | **2762** | 21.9
| ORB | 92 | 102 | 106 | 113 | 109 | 125 | 130 | 129 | 127 | 128 | 1161 | 56
| AKAZE | 166 | 157 | 161 | 155 | 163 | 164 | 173 | 175 | 177 | 179 | **1670** | 7.8
| SIFT | 138 | 132 | 124 | 137 | 134 | 140 | 137 | 148 | 159 | 137 | 1386 | 5.6

### Performance Evaluation 2

Counting the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors.

**Note:** Some combinations of detector and descriptor don't make sense, those results are N/A.

| Detector\Descriptor | BRISK | BRIEF | ORB | FREAK | AKAZE | SIFT |
| --- | --- | --- |--- |--- |--- |--- |
| SHITOMASI | 767 |944|908|768|N/A|927|
| HARRIS | 136|169 |158|140|N/A|160|
| FAST | 2183 |**2831**|**2768**|2233|N/A|**2782**|
| BRISK | 1570 | 1704 |1514|1524|N/A| 1646 |
| ORB | 751 |545|763|420|N/A|763|
| AKAZE | 1215 |1266|1182|1187|1259|1270|
| SIFT | 592|702|N/A|593|N/A|800|

### Performance Evaluation 3

Logging the average time it takes for the whole algorithm (keypoint detection, descriptor extraction, and matching) for all 10 images. 

**Note:** Some combinations of detector and descriptor don't make sense, those results are N/A.

| Detector\Descriptor | BRISK | BRIEF | ORB | FREAK | AKAZE | SIFT |
| --- | --- | --- |--- |--- |--- |--- |
| SHITOMASI| 423.092 |26.577|27.878|72.94|N/A| 48.671|
| HARRIS | 435.698|28.205 |29.37|69.889| N/A| 47.846|
| FAST| 404.842 | **15.89** | **16.105** |64.895|N/A|69.351|
| BRISK| 833.502 |444.389|449.174|491.887|N/A|505.242|
| ORB| 405.005 |**16.984**|22.021|64.095|N/A|88.312|
| AKAZE| 503.094|118.413 |119.24|158.12|191.848|133.906|
| SIFT| 515.823 |155.901|N/A|202.017|N/A|210.767|

### Top 3 Detectors 

**Total Keypoints:** The total number of keypoints on the preceding vehicle for the 10 frames.
**Average Matches:** The average number of matched keypoints on the preceding vehicle between the 10 frames.
**Average Time:** The average time of the whole algorithm for the 10 frames.

| No. | Detector | Descriptor | Total Keypoints | Average Matches | Average Time (ms) |
|------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 | FAST | BRIEF | 4094 | 2831 | 15.89 ms |
| 2 | FAST | ORB | 4094 | 2768 | 16.105 ms |
| 3 | ORB | BRIEF | 1161 | 545 | 16.984 ms |
