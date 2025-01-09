from vedo import Plotter, Sphere, Video

# Create a Video object
# video = Video(name="output_video.mp4", fps=30)

# Set up the Plotter in offscreen mode
plotter = Plotter()  # Plotter(offscreen=False)

# Create a scene and animate it
sphere = Sphere()  # Create the sphere once
for i in range(360):
    plotter.clear()  # Clear previous frame
    sphere.rotate(axis=[0, 0, 1], angle=1)  # Rotate the sphere 1 degree about the Z-axis
    plotter.add(sphere)  # Add the sphere to the plotter
    # Render the scene
    # plotter.show(resetcam=i == 0)  # Reset the camera on the first frame
    plotter.show()  # Reset the camera on the first frame
    # Add the frame to the video
    # video.addFrame()

# Close the Video object
# video.close()

print("Video saved as output_video.mp4")
