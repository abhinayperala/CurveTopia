import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from io import BytesIO

# Function to read CSV and parse paths
def read_csv(csv_file):
    np_path_XYs = np.genfromtxt(csv_file, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Function to plot paths
def plot_polylines(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(4, 4))
    for i, XYs in enumerate(paths_XYs):
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], linewidth=2, label=f'Polyline {i}')
    ax.set_aspect("equal")
    ax.axis('off')
    return fig

# Function to get shape name based on number of vertices
def get_shape_name(approx):
    if len(approx) == 2:
        return "line"
    elif len(approx) == 3:
        return "triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return "square"
        else:
            return "rectangle"
    elif len(approx) > 12:
        return "circle"
    else:
        return "polygon"

# Function to fit a line to points
def fit_line(points):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

# Function to draw shapes on an image
def draw_shape(image, shape_type, points):
    if shape_type == "line":
        pt1 = tuple(points[0][0])
        pt2 = tuple(points[1][0])
        cv2.line(image, pt1, pt2, 0, 2)
    elif shape_type == "rectangle":
        x, y, w, h = cv2.boundingRect(points)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(image, top_left, bottom_right, 0, 2)
    elif shape_type == "circle":
        (x, y), radius = cv2.minEnclosingCircle(points)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, 0, 2)
    elif shape_type == "polygon":
        cv2.polylines(image, [points], isClosed=True, color=0, thickness=2)

def main():
    st.title("CURVETOPIA: A Journey into the World of Curves")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        paths = read_csv(uploaded_file)

        # Plot polylines
        fig = plot_polylines(paths)
        st.pyplot(fig)

        # Save temporary image
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        img = plt.imread(buffer)
        plt.close(fig)

        # Convert image to BGR format
        img_bgr = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

        # Process the image
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((4, 4), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shape_image = np.ones_like(img_bgr) * 255

        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            shape_name = get_shape_name(approx)

            if shape_name == 'line':
                points = np.vstack([point[0] for point in approx])
                slope, intercept = fit_line(points)
                draw_shape(shape_image, 'line', points)
            elif shape_name == 'rectangle':
                draw_shape(shape_image, 'rectangle', approx)
            elif shape_name == 'circle':
                draw_shape(shape_image, 'circle', approx)
            elif shape_name == 'polygon':
                draw_shape(shape_image, 'polygon', approx)

        # Display the output images
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption='Input shapes from CSV')
        st.image(shape_image, caption='Regularized shapes (Final Output)')

    # Footer with credit information
    st.markdown(
        """
        <div style="text-align: center; font-size: 14px; color: gray;">
            Created by Abhinay and Vishal
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
