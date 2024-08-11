# CURVETOPIA: A Journey into the World of Curves

**Created by Abhinay and Vishal**

## Overview

CURVETOPIA is a Streamlit-based web application that allows users to upload CSV files containing paths or shapes defined by coordinates. The application processes these paths to detect geometric shapes, regularizes them, and displays the results on the web interface. This project leverages Python libraries such as OpenCV for image processing, matplotlib for plotting, and scikit-learn for fitting lines to data points.

## Features

- **CSV Upload**: Users can upload CSV files that contain paths defined by coordinates.
- **Shape Detection**: The app detects various geometric shapes such as lines, rectangles, squares, circles, and polygons from the uploaded paths.
- **Regularization**: Detected shapes are regularized (straightened, aligned) and drawn on a new image.
- **Visualization**: The input shapes and regularized shapes are displayed on the web interface.
- **User Interface**: The app provides an easy-to-use interface with a simple file uploader and image display.
- **Footer Credit**: The web app includes a footer with credit information.

## How It Works

1. **Reading the CSV**: The application reads the uploaded CSV file to extract paths defined by coordinate points. Each path represents a potential shape.

2. **Plotting Polylines**: The extracted paths are initially plotted as polylines using matplotlib, and the plot is saved as a temporary image.

3. **Image Processing**: The temporary image is processed using OpenCV. It is converted to grayscale, and edges are detected using the Canny edge detection algorithm. Morphological operations are applied to close gaps in the detected edges.

4. **Contour Detection**: Contours are identified in the processed image. Each contour corresponds to a potential shape in the original paths.

5. **Shape Identification**: The contours are approximated to polygons, and based on the number of vertices, they are classified as lines, rectangles, squares, circles, or polygons.

6. **Shape Drawing**: The identified shapes are drawn on a blank image, creating a regularized version of the input paths. This image is displayed on the web interface.

7. **Web Interface**: The entire process is encapsulated in a Streamlit web app that allows users to upload their CSV files, process them, and view the results in a visually appealing format.

## Setup and Deployment

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/curvetopia.git
   cd curvetopia
2. Install the required dependencies
   ```bash
   pip install -r requirements.txt
3. Run the streamlit app
   ```bash
   streamlit run app.py
