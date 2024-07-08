import cv2
import numpy as np

# colors to be tracked in HSV format
color_ranges = {
    'red': [(0, 120, 70), (10, 255, 255)],
    'green': [(36, 25, 25), (86, 255, 255)],
    'blue': [(94, 80, 2), (126, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'white': [(0, 0, 200), (180, 30, 255)]
}

#quadrants
quadrants = {
    1: [(0, 0), (320, 240)],
    2: [(320, 0), (640, 240)],
    3: [(0, 240), (320, 480)],
    4: [(320, 240), (640, 480)],
}


def is_inside_quadrant(x, y, quadrant):
    (x1, y1), (x2, y2) = quadrant
    return x1 <= x <= x2 and y1 <= y <= y2

# Function to detect and track the balls
def track_balls(video_path, output_video_path, output_text_path):
    cap = cv2.VideoCapture(video_path)
    event_log = []
    ball_last_positions = {color: None for color in color_ranges}
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)

        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 10:  # Filter out small contours
                    center = (int(x), int(y))
                    cv2.circle(frame, center, int(radius), (0, 255, 0), 2)

                    current_quadrant = None
                    for quadrant_id, quadrant in quadrants.items():
                        if is_inside_quadrant(center[0], center[1], quadrant):
                            current_quadrant = quadrant_id
                            break

                    if ball_last_positions[color] is None:
                        event_log.append((frame_time, current_quadrant, color, "Entry"))
                        cv2.putText(frame, f"{color.capitalize()} Entry at Q{current_quadrant} {frame_time:.0f}ms",
                                    (center[0], center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    elif ball_last_positions[color] != current_quadrant:
                        event_log.append((frame_time, ball_last_positions[color], color, "Exit"))
                        event_log.append((frame_time, current_quadrant, color, "Entry"))
                        cv2.putText(frame, f"{color.capitalize()} Exit from Q{ball_last_positions[color]} {frame_time:.0f}ms",
                                    (center[0], center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, f"{color.capitalize()} Entry at Q{current_quadrant} {frame_time:.0f}ms",
                                    (center[0], center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    ball_last_positions[color] = current_quadrant

        out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(output_text_path, 'w') as f:
        for event in event_log:
            f.write(f"{event[0]:.0f}, {event[1]}, {event[2]}, {event[3]}\n")

    return event_log


video_path = 'Assign.mp4'
output_video_path = 'processed_video.avi'
output_text_path = 'event_log.txt'
events = track_balls(video_path, output_video_path, output_text_path)

for event in events:
    print(f"Time: {event[0]:.0f} ms, Quadrant: {event[1]}, Ball Color: {event[2]}, Event: {event[3]}")
