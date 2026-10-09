from math import dist


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [61, 13, 291, 14, 78, 308, 82, 312]


def landmark_to_pixel(landmark, frame_width, frame_height):
    return (
        landmark.x * frame_width,
        landmark.y * frame_height,
    )


def calculate_ear(landmarks, eye_indices, frame_width, frame_height):
    points = [
        landmark_to_pixel(landmarks[index], frame_width, frame_height)
        for index in eye_indices
    ]

    horizontal = dist(points[0], points[3])

    if horizontal == 0:
        return 0.0

    vertical_1 = dist(points[1], points[5])
    vertical_2 = dist(points[2], points[4])

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks, frame_width, frame_height):
    points = [
        landmark_to_pixel(landmarks[index], frame_width, frame_height)
        for index in MOUTH
    ]

    mouth_width = dist(points[0], points[2])

    if mouth_width == 0:
        return 0.0

    mouth_open_1 = dist(points[1], points[3])
    mouth_open_2 = dist(points[4], points[6])
    mouth_open_3 = dist(points[5], points[7])

    return (mouth_open_1 + mouth_open_2 + mouth_open_3) / (3.0 * mouth_width)


def calculate_eye_metrics(landmarks, frame_width, frame_height):
    left_ear = calculate_ear(
        landmarks,
        LEFT_EYE,
        frame_width,
        frame_height,
    )

    right_ear = calculate_ear(
        landmarks,
        RIGHT_EYE,
        frame_width,
        frame_height,
    )

    average_ear = (left_ear + right_ear) / 2.0

    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "ear": average_ear,
    }