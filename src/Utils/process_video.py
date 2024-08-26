import cv2

class VideoProcessing:
    def __init__(self, video_path):
        """
        Initialize the VideoProcessing class with a path to a video file.

        Parameters:
        - video_path (str): The path to the video file.
        """
        self.video_path = video_path

    def extract_all_frames(self, resize=(640, 480)):
        """
        Extracts all frames from the specified video, optionally resizing them to the specified dimensions,
        and returns them as a list.

        Parameters:
        - resize (tuple or None): A tuple (width, height) to resize frames to. 
                                  If None, frames are not resized. Default is (640, 480).

        Returns:
        - frames (list): A list of numpy.ndarray objects, each representing a frame.
        """
        # Initialize a video capture object with the video file
        cap = cv2.VideoCapture(self.video_path)
        frames = []

        # Check if the video was opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            return []

        # Loop through the video frame by frame
        while True:
            ret, frame = cap.read()  # Read a single frame
            if not ret:
                break  # Exit the loop if no frame is returned (end of video)
            
            # Resize the frame if a resize dimension is specified
            if resize is not None:
                frame = cv2.resize(frame, resize)
            
            frames.append(frame)  # Append the extracted (and possibly resized) frame to the list

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

        return frames

# Example usage
if __name__ == "__main__":
    video_path = '../videos/vid1.mp4'
    video_processor = VideoProcessing(video_path)

    # Extract frames with default resizing (640x480)
    frames_resized = video_processor.extract_all_frames()

    # Extract frames without resizing
    frames_original = video_processor.extract_all_frames(resize=None)

    print(f"Extracted {len(frames_resized)} resized frames.")
    print(f"Extracted {len(frames_original)} original frames.")
