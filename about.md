ML OUTPUTS:

1. Spatial Object Detection (Fire and Smoke Perception)
    The Process: The system utilizes a trained machine learning model to scan every incoming video frame in real time, detecting and isolating hazardous regions.

    Key Terms & Definitions:
        YOLO (You Only Look Once): A state-of-the-art real-time object detection algorithm. It is called "You Only Look Once" because it processes the entire image in a single forward pass through the neural network to locate objects instantly.
        
        Confidence Percentage: A statistical probability score (e.g., $85\%$) outputted by the model indicating how certain it is that the detected object is actually fire or smoke.
        
        Bounding Box Coordinates: The spatial coordinates ($[x_1, y_1, x_2, y_2]$) that define the rectangular border enclosing the detected hazard on the screen, mapping where it is located.

2. Visual Severity Classification (Intensity Tiering)
    The Process: Whenever a fire bounding box is detected, the system dynamically crops that specific area from the raw frame. This crop is passed to a secondary classifier trained on a custom dataset split into multiple category folders to classify the intensity.

    Key Terms & Definitions:

        ROI (Region of Interest): A cropped, localized sub-section of an image or video frame—in this case, just the pixels containing the fire—so the system can analyze it closely without processing unnecessary background scenery.

        Severity Tiers (Mild, Moderate, Severe): Categorical labels determining the visual intensity of the combustion based on flame brightness, structural spread, and smoke density.

        CNN (Convolutional Neural Network): A class of deep learning neural networks specialized for processing grid-like data, such as images, by extracting visual patterns (like edges, colors, and textures).

3. Temporal Behavior Tracking (Kinematic Trend Analysis)
    The Process: The system maintains a rolling history buffer tracking hazard frames over time (e.g., a window of 15 frames). It measures how the physical pixel footprint changes across frames and applies mathematical regression to determine the trajectory.

    Key Terms & Definitions:

        Temporal History Buffer: A temporary memory queue in code that stores data from recent consecutive frames to analyze how a hazard changes over time rather than just looking at a single snapshot.

        Kinematic Tracking: Monitoring movement, expansion, or physical changes of an object over time.

        Linear Regression Slope (np.polyfit): A mathematical curve-fitting method used to find the general "line of best fit" for pixel growth over time. If the slope is positive, the hazard is expanding; if negative, it is shrinking.

        Behavior Status (Growing, Stable, Diminishing): The operational trend outputted by the slope analysis, indicating whether a fire front or smoke plume is escalating, holding steady, or dying down.

4. Risk / Priority Scoring Module (Decision Support Triage)
    The Process: Instead of leaving responders to look at raw numbers separately, this module fuses the detection confidence, bounding box pixel area (footprint size), severity tier, and kinematic behavior trend into a single mathematical formula to generate an overall sector urgency score from 0 to 100.

    Key Terms & Definitions:

        Decision-Support / Triage Layer: Software functionality that filters, ranks, and prioritizes data so human operators can immediately see which emergency requires the fastest intervention.

        Heuristic Decision-Fusion: Combining multiple disparate data metrics (confidence + size + severity + growth rate) using transparent, logical rules to arrive at a single actionable priority score.

        Urgency Index (0–100 Scale): A standardized rating system where low scores represent minor, manageable events, and high scores (e.g., >75) trigger critical emergency dispatch alerts for multi-point wildfires.