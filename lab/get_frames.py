# Code for getting frames from the webcam and apply a function to them every X frames.
import cv2


def process_frame(frame):
    """
    Example function to process a frame.
    This function can be modified to perform any operation on the frame.
    :param frame: The frame to process.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the processed frame
    # cv2.imshow('Processed Frame', gray_frame)



def get_frames(func, every=100):
    """
    Get frames from the webcam and apply a function to them every X frames.
    :param func: Function to apply to the frames.
    :param every: Apply the function every X frames.
    """
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % every == 0:
            process_frame(frame)

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage: process every 10th frame
    get_frames(process_frame, every=100)