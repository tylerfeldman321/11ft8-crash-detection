## Sign Detection
Code for detecting whether the "OVERHEIGHT MUST TURN" sign is on.

`sign_detector.py` - Contains a `SignDetector` class that can process videos to extract template matching results and sliding window features from the template matching results.
`data_collection.py` - Contains functions to collect template matching data. This includes labeling the footage, extracting templates, plotting template matching results, and plotting the KDE and ROC to evaluate the template matching results.
`utils.py` - Contains utility functions.
`templates/` - Contains templates to use for template matching.
