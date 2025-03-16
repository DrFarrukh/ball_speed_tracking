import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import subprocess

def convert_dav_to_mp4(dav_path, output_path=None):
    """
    Convert a .dav file to .mp4 format using FFmpeg.
    
    Parameters:
    - dav_path: Path to the .dav file
    - output_path: Path for the output .mp4 file (if None, will use the same name with .mp4 extension)
    
    Returns:
    - Path to the converted MP4 file
    """
    if output_path is None:
        # Use the same name but with .mp4 extension
        output_path = os.path.splitext(dav_path)[0] + '.mp4'
    
    try:
        # Check if FFmpeg is installed
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Convert the file
        command = [
            'ffmpeg',
            '-i', dav_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path
        ]
        
        print("Converting DAV to MP4...")
        subprocess.run(command, check=True)
        print(f"Conversion complete: {output_path}")
        
        return output_path
    
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in the system PATH.")
        print("Please install FFmpeg (https://ffmpeg.org/) and try again.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        return None

def try_open_dav_direct(dav_path):
    """
    Attempt to open a .dav file directly with OpenCV.
    Returns the video capture object if successful, None otherwise.
    """
    cap = cv2.VideoCapture(dav_path)
    if cap.isOpened():
        print("Successfully opened .dav file directly with OpenCV")
        return cap
    else:
        print("Could not open .dav file directly. Will try conversion.")
        return None

def calculate_ball_speed(video_path, known_distance_meters, calibration_points=None, frame_start=0, frame_end=None):
    """
    Calculate the speed of a cricket ball from video footage.
    Works with .dav files by either opening them directly or converting to MP4.
    
    Parameters:
    - video_path: Path to the video file (.dav or other formats)
    - known_distance_meters: A known distance in the video in meters (e.g., pitch length = 20.12m)
    - calibration_points: List of tuples [(x1,y1), (x2,y2)] representing points in the image that are known_distance_meters apart
    - frame_start: Frame to start analysis from
    - frame_end: Frame to end analysis at (if None, will use interactive selection)
    
    Returns:
    - speed_mps: Speed in meters per second
    - speed_kph: Speed in kilometers per hour
    """
    # Check if it's a .dav file
    is_dav = video_path.lower().endswith('.dav')
    
    if is_dav:
        # Try to open the .dav file directly
        cap = try_open_dav_direct(video_path)
        
        # If direct opening failed, try conversion
        if cap is None:
            mp4_path = convert_dav_to_mp4(video_path)
            if mp4_path is None:
                raise Exception("Failed to open or convert .dav file")
            video_path = mp4_path
            cap = cv2.VideoCapture(video_path)
    else:
        # Not a .dav file, open normally
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video loaded: {frame_width}x{frame_height} at {fps} fps")
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    
    # If calibration points not provided, use interactive selection
    if calibration_points is None:
        print("Please select two points that are known_distance_meters apart.")
        print("For cricket, this could be the length of the pitch (20.12m) or the distance between wickets.")
        calibration_points = select_calibration_points(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)  # Reset to start frame
    
    # Calculate pixels per meter
    pixels_per_meter = calculate_pixels_per_meter(calibration_points, known_distance_meters)
    print(f"Calibration: {pixels_per_meter:.2f} pixels per meter")
    
    # Track the ball
    print("\nBall tracking instructions:")
    print("  - Press 'b' to enter ball marking mode for the current frame")
    print("  - Click on the ball's position in the frame")
    print("  - Press 'n' to move to the next frame without marking")
    print("  - Press 'q' to finish tracking and calculate speed")
    
    ball_positions, frame_numbers = track_ball(cap, frame_start, frame_end)
    
    if not ball_positions or len(ball_positions) < 2:
        print("Could not track the ball for enough frames.")
        return None, None
    
    # Calculate distances between consecutive positions
    distances_pixels = []
    for i in range(1, len(ball_positions)):
        x1, y1 = ball_positions[i-1]
        x2, y2 = ball_positions[i]
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        distances_pixels.append(distance)
    
    # Convert distances to meters
    distances_meters = [d / pixels_per_meter for d in distances_pixels]
    
    # Calculate time intervals between frames
    time_intervals = [(frame_numbers[i] - frame_numbers[i-1]) / fps for i in range(1, len(frame_numbers))]
    
    # Calculate instantaneous speeds
    speeds = [dist / time for dist, time in zip(distances_meters, time_intervals)]
    
    # Calculate average speed
    total_distance_meters = sum(distances_meters)
    total_time_seconds = (frame_numbers[-1] - frame_numbers[0]) / fps
    avg_speed_mps = total_distance_meters / total_time_seconds
    avg_speed_kph = avg_speed_mps * 3.6  # Convert m/s to km/h
    
    # Calculate maximum speed
    max_speed_mps = max(speeds) if speeds else 0
    max_speed_kph = max_speed_mps * 3.6
    
    # Plot the trajectory and speed
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot trajectory
    xs = [p[0] for p in ball_positions]
    ys = [p[1] for p in ball_positions]
    
    ax1.plot(xs, ys, 'bo-', label='Ball positions')
    ax1.invert_yaxis()  # Invert y-axis to match image coordinates
    ax1.set_title('Ball Trajectory')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.grid(True)
    
    # Fit a polynomial to the trajectory
    if len(xs) > 2:
        try:
            t = np.arange(len(xs))
            
            # Determine appropriate polynomial degree (less than number of points)
            poly_degree = min(2, len(xs) - 1)
            
            px = np.polyfit(t, xs, poly_degree)
            py = np.polyfit(t, ys, poly_degree)
            
            t_smooth = np.linspace(0, len(xs)-1, 100)
            x_smooth = np.polyval(px, t_smooth)
            y_smooth = np.polyval(py, t_smooth)
            
            ax1.plot(x_smooth, y_smooth, 'r-', label='Fitted trajectory')
            ax1.legend()
        except Exception as e:
            print(f"Could not fit polynomial to trajectory: {e}")
    
    # Plot speeds
    frame_indices = list(range(1, len(speeds) + 1))
    ax2.bar(frame_indices, [s * 3.6 for s in speeds], color='green')
    ax2.axhline(y=avg_speed_kph, color='r', linestyle='-', label=f'Avg: {avg_speed_kph:.1f} km/h')
    ax2.set_title('Ball Speed between Frames')
    ax2.set_xlabel('Frame Pair')
    ax2.set_ylabel('Speed (km/h)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('ball_speed_analysis.png')
    plt.show()
    
    print(f"\nResults:")
    print(f"Average ball speed: {avg_speed_mps:.2f} m/s ({avg_speed_kph:.2f} km/h)")
    print(f"Maximum ball speed: {max_speed_mps:.2f} m/s ({max_speed_kph:.2f} km/h)")
    
    # Return average speed
    return avg_speed_mps, avg_speed_kph

def select_calibration_points(cap):
    """Interactive selection of calibration points."""
    # Get a frame to display
    ret, frame = cap.read()
    if not ret:
        raise Exception("Could not read frame for calibration")
    
    # Create a copy of the frame to draw on
    img = frame.copy()
    
    # List to store the selected points
    points = []
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(img, points[0], points[1], (0, 255, 0), 2)
                distance_pixels = np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
                cv2.putText(img, f"Distance: {distance_pixels:.1f} px", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Calibration', img)
    
    # Display the image and wait for mouse clicks
    cv2.imshow('Calibration', img)
    cv2.setMouseCallback('Calibration', mouse_callback)
    
    print("Please click on two points that represent a known distance (e.g., the pitch length)")
    print("Press any key after selecting two points")
    
    while len(points) < 2:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.waitKey(1000)  # Brief pause to see the selection
    cv2.destroyWindow('Calibration')
    
    return points

def calculate_pixels_per_meter(points, known_distance_meters):
    """Calculate pixels per meter based on calibration points."""
    x1, y1 = points[0]
    x2, y2 = points[1]
    pixel_distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    pixels_per_meter = pixel_distance / known_distance_meters
    return pixels_per_meter

def track_ball(cap, frame_start, frame_end):
    """Track the ball across multiple frames with clear user instructions."""
    # Lists to store ball positions and frame numbers
    ball_positions = []
    frame_numbers = []
    
    # Current frame number
    current_frame = frame_start
    
    # Font settings for displaying frame information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 0)
    font_thickness = 2
    
    # Define window name
    window_name = 'Ball Tracking'
    cv2.namedWindow(window_name)
    
    # Process frames until end or until user stops
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret or (frame_end is not None and current_frame >= frame_end):
            break
        
        # Create a copy to draw on
        display_frame = frame.copy()
        
        # Add frame number and instructions
        cv2.putText(display_frame, f"Frame: {current_frame}", 
                   (10, 30), font, font_scale, font_color, font_thickness)
        cv2.putText(display_frame, "Press 'b' to mark ball, 'n' for next frame, 'q' to finish", 
                   (10, 60), font, font_scale, font_color, font_thickness)
        
        # Display marked positions on the current frame
        for i, pos in enumerate(ball_positions):
            cv2.circle(display_frame, pos, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i+1), 
                       (pos[0]+10, pos[1]), font, font_scale, (0, 0, 255), font_thickness)
        
        # Display the frame
        cv2.imshow(window_name, display_frame)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        # If 'q' is pressed, stop tracking
        if key == ord('q'):
            break
        
        # If 'n' is pressed, go to next frame without marking
        elif key == ord('n'):
            current_frame += 1
            continue
        
        # If 'b' is pressed, mark the ball position
        elif key == ord('b'):
            # Create a copy for ball marking
            marking_frame = display_frame.copy()
            cv2.putText(marking_frame, "Click on the ball position", 
                       (10, 90), font, font_scale, (0, 255, 255), font_thickness)
            cv2.imshow(window_name, marking_frame)
            
            # Set up mouse callback for marking
            mark_done = [False]  # Use list to modify in callback
            ball_marked = [False]
            clicked_pos = [None]
            
            def mark_ball(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and not ball_marked[0]:
                    clicked_pos[0] = (x, y)
                    ball_marked[0] = True
                    
                    # Draw the marked position
                    cv2.circle(marking_frame, (x, y), 5, (0, 0, 255), -1)
                    idx = len(ball_positions) + 1
                    cv2.putText(marking_frame, str(idx), 
                               (x+10, y), font, font_scale, (0, 0, 255), font_thickness)
                    
                    cv2.putText(marking_frame, "Press Enter to confirm or ESC to try again", 
                               (10, 120), font, font_scale, (0, 255, 255), font_thickness)
                    cv2.imshow(window_name, marking_frame)
            
            # Set the mouse callback
            cv2.setMouseCallback(window_name, mark_ball)
            
            # Wait for ball marking or cancellation
            while not mark_done[0]:
                key = cv2.waitKey(0) & 0xFF
                
                # Enter key confirms the marking
                if key == 13 and ball_marked[0]:  # Enter key
                    ball_positions.append(clicked_pos[0])
                    frame_numbers.append(current_frame)
                    mark_done[0] = True
                
                # ESC key cancels the marking
                elif key == 27:  # ESC key
                    # Reset and try again
                    marking_frame = display_frame.copy()
                    cv2.putText(marking_frame, "Click on the ball position", 
                               (10, 90), font, font_scale, (0, 255, 255), font_thickness)
                    cv2.imshow(window_name, marking_frame)
                    ball_marked[0] = False
            
            # Move to next frame after marking
            current_frame += 1
            
            # Reset the mouse callback
            cv2.setMouseCallback(window_name, lambda *args: None)
        
        else:
            # Any other key, just move to the next frame
            current_frame += 1
    
    cv2.destroyAllWindows()
    
    if len(ball_positions) < 2:
        print("Warning: Less than 2 ball positions were marked. Cannot calculate speed.")
    
    return ball_positions, frame_numbers

def main():
    """Main function to run the ball speed calculation."""
    print("\n=== Cricket Ball Speed Measurement ===\n")
    
    # Get the video file path
    video_path = input("Enter the path to your .dav video file: ")
    
    # Check if the file exists
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return
    
    # Get the known distance
    try:
        known_distance = float(input("Enter a known distance in meters (e.g., 20.12 for cricket pitch): "))
    except ValueError:
        print("Error: Please enter a valid number for the distance.")
        known_distance = 20.12  # Default to cricket pitch length
        print(f"Using default value: {known_distance} meters")
    
    # Get start frame (optional)
    try:
        start_frame = int(input("Enter starting frame number (default: 0): ") or "0")
    except ValueError:
        print("Invalid frame number. Using default: 0")
        start_frame = 0
    
    # Calculate ball speed
    try:
        speed_mps, speed_kph = calculate_ball_speed(
            video_path,
            known_distance,
            frame_start=start_frame
        )
        
        if speed_mps is not None:
            print("\nAnalysis completed successfully!")
            print(f"Results saved as 'ball_speed_analysis.png'\n")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()