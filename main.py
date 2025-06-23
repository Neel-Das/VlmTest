import torch
import cv2
import serial
import time
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT import
import requests
from io import BytesIO
import json

max_distance = 3000.0


def clamp_depth(depth_image, max_distance_mm=max_distance):
    """Clamp depth values to a maximum distance and convert to uint16."""
    depth_clamped = np.where(depth_image > max_distance_mm, max_distance_mm, depth_image)
    return depth_clamped.astype(np.uint16)


# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the YOLO model
model = YOLO("C:/Users/rupsd/Downloads/integration_code-master/integration_code-master/src/best.pt").to(device)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get depth scale for accurate depth measurements
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale: {depth_scale}")

# Post-processing filters
spatial_filter = rs.spatial_filter()
spatial_filter.set_option(rs.option.filter_magnitude, 3)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.6)
spatial_filter.set_option(rs.option.filter_smooth_delta, 10)

temporal_filter = rs.temporal_filter()
temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.6)
temporal_filter.set_option(rs.option.filter_smooth_delta, 10)

hole_filling_filter = rs.hole_filling_filter()

align_to = rs.stream.color
align = rs.align(align_to)

# Class ID for 'Lid_v3' (adjust for your custom model)
LID_CLASS_ID = 0

# Directory to store tensors
tensor_dir = './tensor_data/'
comb_tensor_dir = "./comb_tensor_data/"
os.makedirs(tensor_dir, exist_ok=True)

# Flag for one-time output
output_done = False

# Lists to store data across frames
all_yolo_boundaries = []
all_confidence_scores = []
all_rgb_values = []
all_depth_vectors = []

print("Press 'q' to quit.")

# Initialize Serial Communication
arduino_port = serial.Serial('COM6', 9600)
time.sleep(2)  # Allow Arduino to initialize

# Initialize DeepSORT tracker with correct parameters
deepsort = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.2,
    nn_budget=100
)


def send_data_to_arduino(serial_port, x, y):
    """Sends x, y coordinates to Arduino via serial."""
    data_string = f"{x - 14.7638:.2f},{y:.2f}\n"  # Format the data
    try:
        serial_port.write(data_string.encode())  # Encode and send
        print(f"Sent to Arduino: {data_string.strip()}")
    except serial.SerialException as e:
        print(f"Error sending data to Arduino: {e}")


try:
    while True:
        # Capture and align frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Apply filters to the depth frame
        depth_frame = spatial_filter.process(depth_frame)
        depth_frame = temporal_filter.process(depth_frame)
        depth_frame = hole_filling_filter.process(depth_frame)

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())  # BGR format

        # Clamp depth values
        depth_clamped = clamp_depth(depth_image, max_distance_mm=max_distance)

        # Verify frame alignment and dimensions
        if depth_clamped.shape != frame.shape[:2]:
            print(f"Warning: Depth ({depth_clamped.shape}) and color ({frame.shape[:2]}) shape mismatch!")
            continue

        # Run YOLO inference
        results = model(frame, device=device, conf=0.94)
        annotated_frame = results[0].plot(labels=True, masks=True, boxes=True)
        det = results[0]

        # Prepare detections for DeepSORT
        detections = []
        if det.boxes is not None and len(det.boxes.xyxy) > 0:
            for i in range(len(det.boxes.xyxy)):
                if int(det.boxes.cls[i]) != LID_CLASS_ID:
                    continue
                conf = float(det.boxes.conf[i])
                if conf <= 0.9:
                    continue
                x_min, y_min, x_max, y_max = map(float, det.boxes.xyxy[i].cpu().numpy())
                w = x_max - x_min
                h = y_max - y_min
                detections.append(([x_min, y_min, w, h], conf, LID_CLASS_ID))

        # Update DeepSORT tracker
        tracks = deepsort.update_tracks(detections, frame=frame)

        # Initialize lists to store raw data for this frame
        yolo_boundaries = []
        confidence_scores = []
        rgb_values = []
        depth_vectors = []

        # Build union mask for visualization and depth extraction
        union_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Process detections
        if det.masks is not None and len(det.masks.data) > 0 and len(det.boxes.cls) > 0:
            # First pass: Compute union mask and full depth map
            masked_depth_full = np.zeros_like(depth_clamped, dtype=np.uint16)
            for i in range(len(det.masks.data)):
                mask_tensor = det.masks.data[i]
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                union_mask = np.where(mask_np == 1, 1, union_mask)
                masked_depth_full = np.where(mask_np == 1, depth_clamped, masked_depth_full)

            # Second pass: Process each detection and add text with ID
            for i in range(len(det.masks.data)):
                try:
                    # Verify index and class
                    if i >= len(det.boxes.cls) or i >= len(det.boxes.xyxy) or i >= len(det.boxes.conf):
                        print(f"Warning: Invalid index {i} for detection {i + 1}")
                        continue
                    if int(det.boxes.cls[i]) != LID_CLASS_ID:
                        continue

                    # Extract confidence score
                    conf = det.boxes.conf[i].cpu().numpy()
                    conf_value = float(conf)
                    if conf_value <= 0.9:
                        print(f"Warning: Confidence {conf_value} below 0.9 for detection {i + 1}, skipping")
                        continue

                    # Extract segmentation mask
                    mask_tensor = det.masks.data[i]
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)

                    # Compute segmentation boundaries
                    y_indices, x_indices = np.where(mask_np == 1)
                    if len(y_indices) == 0 or len(x_indices) == 0:
                        print(f"Warning: Empty mask for detection {i + 1}")
                        continue
                    y_min, y_max = y_indices.min(), y_indices.max()
                    x_min, x_max = x_indices.min(), x_indices.max()
                    seg_boundary = np.array([x_min, y_min, x_max, y_max])

                    # Calculate center of the bounding box
                    xCOOL = int((x_min + x_max) / 2)
                    yCOOL = int((y_min + y_max) / 2)

                    # Pixel to robot movement
                    rX = yCOOL - 240
                    rY = xCOOL - 320

                    # mm to in
                    inX = (rX / 10) / 2.54
                    inY = (rY / 10) / 2.54

                    data_string = f"{inX},{inY}\n"

                    if cv2.waitKey(1) == ord('g'):
                        send_data_to_arduino(arduino_port, inX, inY)

                    # Find matching track ID
                    track_id = -1
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        tlwh = track.to_tlwh()
                        track_xmin, track_ymin, track_w, track_h = tlwh
                        track_xmax = track_xmin + track_w
                        track_ymax = track_ymin + track_h
                        # Check if the track matches the current detection (based on proximity)
                        if (abs(x_min - track_xmin) < 10 and abs(y_min - track_ymin) < 10 and
                                abs(x_max - track_xmax) < 10 and abs(y_max - track_ymax) < 10):
                            track_id = track.track_id
                            break

                    # Check if track_id is odd
                    if track_id % 2 == 1:
                        # Prepare stats dictionary
                        stats = {
                            "yolo_boundary": yolo_boundaries[i].tolist(),
                            "confidence": float(confidence_scores[i]),
                            "rgb_values": rgb_values[i].tolist(),
                            "depth_vector": depth_vectors[i].tolist()
                        }

                        # Send to VLM
                        vlm_response = send_to_vlm(frame, yolo_boundaries[i], stats)
                        print(f"VLM Response for Track ID {track_id}: {vlm_response}")

                        # Optionally, save the response (e.g., to a file or MongoDB later)
                        with open(f"vlm_responses/track_{track_id}_{timestamp}.json", "w") as f:
                            json.dump(vlm_response, f)

                    # Add text with ID and coordinates above the bounding box
                    text = f"ID: {track_id} ({xCOOL}, {yCOOL})"
                    text_position = (x_min, y_min - 10)  # Position text 10 pixels above the top-left corner
                    cv2.putText(
                        annotated_frame,
                        text,
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # Font scale
                        (0, 255, 0),  # Green text
                        1,  # Thickness
                        cv2.LINE_AA
                    )

                    # Compute average RGB values
                    masked_rgb = np.where(mask_np[:, :, np.newaxis], frame, 0)
                    mask_sum = mask_np.sum()
                    if mask_sum == 0:
                        print(f"Warning: Empty mask for detection {i + 1}")
                        continue
                    rgb_mean = masked_rgb.sum(axis=(0, 1)) / mask_sum

                    # Extract depth values within the mask
                    masked_depth = np.where(mask_np == 1, depth_clamped, 0)
                    valid_depths = masked_depth[masked_depth > 0]

                    non_zero_count = np.count_nonzero(masked_depth)
                    print(f"Detection {i + 1} - Non-zero depth values in masked_depth: {non_zero_count}")

                    if non_zero_count == 0:
                        print(f"Warning: No valid depth data for detection {i + 1}, using zeros")
                        depth_vector = np.zeros(8, dtype=np.uint16)
                    else:
                        depth_min = int(np.min(valid_depths))
                        depth_max = int(np.max(valid_depths))
                        depth_mean = int(np.mean(valid_depths))
                        depth_median = int(np.median(valid_depths))
                        p10 = int(np.percentile(valid_depths, 10))
                        p25 = int(np.percentile(valid_depths, 25))
                        p75 = int(np.percentile(valid_depths, 75))
                        p90 = int(np.percentile(valid_depths, 90))
                        depth_vector = np.array([depth_min, p10, p25, depth_median,
                                                 depth_mean, p75, p90, depth_max],
                                                dtype=np.uint16)
                        depth_min_cm = depth_min * depth_scale * 100
                        depth_max_cm = depth_max * depth_scale * 100
                        depth_mean_cm = depth_mean * depth_scale * 100
                        depth_median_cm = depth_median * depth_scale * 100
                        print(
                            f"  Depth stats (cm): Min={depth_min_cm:.2f}, Max={depth_max_cm:.2f}, Mean={depth_mean_cm:.2f}, Median={depth_median_cm:.2f}")
                        print(f"  Depth vector [min,p10,p25,median,mean,p75,p90,max]: {depth_vector}")

                    yolo_boundaries.append(seg_boundary)
                    confidence_scores.append(conf_value)
                    rgb_values.append(rgb_mean)
                    depth_vectors.append(depth_vector)

                    all_yolo_boundaries.append(seg_boundary)
                    all_confidence_scores.append(conf_value)
                    all_rgb_values.append(rgb_mean)
                    all_depth_vectors.append(depth_vector)

                except Exception as e:
                    print(f"Error processing detection {i + 1}: {e}")
                    continue

        # Convert lists to tensors efficiently
        yolo_tensor = torch.tensor(np.array(yolo_boundaries), dtype=torch.float32) if yolo_boundaries else torch.tensor(
            [], dtype=torch.float32)
        conf_tensor = torch.tensor(np.array(confidence_scores),
                                   dtype=torch.float32) if confidence_scores else torch.tensor([], dtype=torch.float32)
        rgb_tensor = torch.tensor(np.array(rgb_values), dtype=torch.float32) if rgb_values else torch.tensor([],
                                                                                                             dtype=torch.float32)
        depth_tensor = torch.tensor(np.array(depth_vectors), dtype=torch.float32) if depth_vectors else torch.tensor([],
                                                                                                                     dtype=torch.float32)
        # yolo_t has shape (N, 4), so yolo_t.shape[0] == N
        feature_list = []
        for i in range(yolo_tensor.shape[0]):
            b = yolo_tensor[i]  # (4,)
            c = conf_tensor[i].unsqueeze(0)  # (1,)
            r = rgb_tensor[i]  # (3,)
            d = depth_tensor[i]  # (8,)
            feat = torch.cat([b, c, r, d], dim=0)  # (16,)
            feature_list.append(feat)

        all_fields = torch.stack(feature_list, dim=0) if feature_list else torch.empty((0, 16))
        vectors = all_fields.cpu().numpy().tolist()
        if all_fields is not None:
            try:
                np.save(f'{comb_tensor_dir}/combined{timestamp}.npy', yolo_tensor.numpy())
            except Exception as e:
                print(f"Error saving data in {comb_tensor_dir}: {e}")

        # One-time output of tensor data
        if not output_done and len(yolo_boundaries) > 0:
            print("\nInitial Detection Tensor Data:")
            print(f"YOLO Boundaries (Segmentation): {yolo_tensor}")
            print(f"Confidence Scores: {conf_tensor}")
            print(f"RGB Values: {rgb_tensor}")
            print(f"Depth Vectors Shape: {depth_tensor.shape}")
            print(f"Depth Vectors: {depth_tensor}")

            # Save as dictionary of numpy arrays instead of trying to save a tensor directly
            tensor_data = {
                'yolo': yolo_tensor.cpu().numpy(),
                'conf': conf_tensor.cpu().numpy(),
                'rgb': rgb_tensor.cpu().numpy(),
                'depth': depth_tensor.cpu().numpy()
            }
            np.save('detection_tensors.npy', tensor_data)
            print("Tensor data saved to 'detection_tensors.npy'")
            output_done = True

        # Print vectors for verification (per frame)
        if yolo_boundaries:
            print(f"\nFrame Data:")
            for i in range(len(yolo_boundaries)):
                print(f"Detection {i + 1}:")
                print(f"  YOLO Boundaries (Segmentation): {yolo_boundaries[i]}")
                print(f"  Confidence: {confidence_scores[i]}")
                print(f"  RGB Values: {rgb_values[i]}")
                print(f"  Depth Vector Shape: {depth_vectors[i].shape}")
                print(f"  Depth Vector: {depth_vectors[i]}")

        # Save tensors to file
        if len(yolo_boundaries) > 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            try:
                # Create a single dictionary with all tensor data for this frame
                frame_data = {
                    'yolo': yolo_tensor.cpu().numpy(),
                    'conf': conf_tensor.cpu().numpy(),
                    'rgb': rgb_tensor.cpu().numpy(),
                    'depth': depth_tensor.cpu().numpy(),
                    'timestamp': timestamp
                }

                # Save as a single file with all data
                np.save(f'{tensor_dir}/frame_data_{timestamp}.npy', frame_data)

                # Also save each tensor separately for compatibility
                np.save(f'{tensor_dir}/yolo_{timestamp}.npy', yolo_tensor.cpu().numpy())
                np.save(f'{tensor_dir}/conf_{timestamp}.npy', conf_tensor.cpu().numpy())
                np.save(f'{tensor_dir}/rgb_{timestamp}.npy', rgb_tensor.cpu().numpy())
                np.save(f'{tensor_dir}/depth_{timestamp}.npy', depth_tensor.cpu().numpy())

                print(f"Saved tensors for frame {timestamp}")
            except Exception as e:
                print(f"Error saving tensors for frame {timestamp}: {e}")
                import traceback

                traceback.print_exc()

        # Visualize depth with improved scaling
        masked_depth = np.where(union_mask, depth_clamped, 0).astype(np.float32)
        valid_depth = masked_depth[masked_depth > 0]
        if valid_depth.size > 0:
            depth_normalized = (masked_depth - valid_depth.min()) / (
                        valid_depth.max() - valid_depth.min() + 1e-10) * 255.0
            depth_normalized = np.clip(depth_normalized, 0, 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(masked_depth, dtype=np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

        cv2.imshow("YOLOv11n-seg RealSense", annotated_frame)
        cv2.imshow("Depth Map (Lid Only)", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if len(all_yolo_boundaries) > 0 and not output_done:
        yolo_tensor = torch.tensor(np.array(all_yolo_boundaries), dtype=torch.float32)
        conf_tensor = torch.tensor(np.array(all_confidence_scores), dtype=torch.float32)
        rgb_tensor = torch.tensor(np.array(all_rgb_values), dtype=torch.float32)
        depth_tensor = torch.tensor(np.array(all_depth_vectors), dtype=torch.float32)

        print("\nFinal Detection Tensor Data (Before Exit):")
        print(f"YOLO Boundaries (Segmentation): {yolo_tensor}")
        print(f"Confidence Scores: {conf_tensor}")
        print(f"RGB Values: {rgb_tensor}")
        print(f"Depth Vectors Shape: {depth_tensor.shape}")
        print(f"Depth Vectors: {depth_tensor}")

        # Create a dictionary of numpy arrays for saving
        tensor_data = {
            'yolo': yolo_tensor.cpu().numpy(),
            'conf': conf_tensor.cpu().numpy(),
            'rgb': rgb_tensor.cpu().numpy(),
            'depth': depth_tensor.cpu().numpy()
        }

        # Save final data
        np.save('detection_tensors_final.npy', tensor_data)
        print("Final tensor data saved to 'detection_tensors_final.npy'")

        # Also save to tensor_dir
        final_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        np.save(f'{tensor_dir}/final_data_{final_timestamp}.npy', tensor_data)
        print(f"Final tensor data also saved to {tensor_dir}/final_data_{final_timestamp}.npy")

    pipeline.stop()
    cv2.destroyAllWindows()
    print("RealSense pipeline stopped.")
